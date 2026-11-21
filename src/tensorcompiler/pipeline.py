"""流水线编排：**从图到产物**。

```
load graphs
   ↓
每个图：完整降级链（逐级漏斗）
   ↓
策略 × 图 的对比矩阵        ← 扇出
   ↓
消融归因                    ← 收拢：每个 pass 的贡献
   ↓
三条独立路径的语义互校 + 静态/动态双成本互校
   ↓
门槛 / 对抗用例 / 设计期望
   ↓
落盘
```

⚠️ 顺序是**有理由的**：对比与归因在**语义核对之前**跑，所以如果某个 pass
改了语义，报告里会先出现「成本很漂亮」再出现「语义不一致」——
**两件事分开报**，而不是让前者把后者盖住。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import attribution as AT
from . import compare as CMP
from . import cost as C
from . import emit as E
from . import expectations as EXP
from . import invariants as INV
from .config import protocol_digest
from .graph import load_graphs
from .interp import (ExecResult, bitwise_equal, eval_graph, execute,
                     max_abs_diff)
from .ir import Graph
from .passes import lower_ladder
from .rng import tensor_rng
from .strategies import enumerate_strategies, strategy_id


@dataclass
class Run:
    """一次完整运行。**渲染器只吃 `products`，不吃这里的任何运行期对象。**"""
    products: dict[str, Any]
    graphs: tuple[Graph, ...]
    out_dir: Path


#: 对抗用例**第一趟**跑之前必须已经存在的产物。
#:
#: ⚠️ 不含 `summary.json`：它在两趟之间才生成（报告要印对抗用例的结论，
#: 而对抗用例里有一条要调 `render` —— 那是一个环，见 `run_pipeline`）。
#: ⚠️ 不含 `adversarial.json` 自己（它是这一层的产物，第二趟之前才放进去）。
#:
#: 缺了不会报错 —— 会静默变成「违反」，而同一批用例在 `check` 里读盘上的
#: 完整产物又是绿的。**两个数字说的是同一件事，却对不上。**
ADVERSARIAL_INPUTS: tuple[str, ...] = (
    "graphs.json", "ladder.json", "compare.json", "attribution.json",
    "semantics.json", "crosscheck.json", "acceptance.json", "expectations.json",
    "config.json", "benchmark.json",
    # ⚠️ `version.json` 的**声明部分**（`versioned_files` / `excluded`）也算输入：
    #    `no_elapsed_time_key_in_the_versioned_products` 靠它才知道该走哪些文件。
    #    少了它，那条用例会拿到空的 `versioned_files` ⇒ **恒绿**
    #    （「没有实测耗时进摘要」在「摘要里没有文件」上当然成立）。
    "version.json",
)


# --------------------------------------------------------------------- 输入
def synth_feeds(g: Graph, cfg: dict[str, Any]) -> dict[str, list[list[float]]]:
    """给一张图合成确定性输入（`input` 与 `const` 角色都需要有值）。"""
    out: dict[str, list[list[float]]] = {}
    for t in g.tensors:
        if t.role not in ("input", "const"):
            continue
        r = tensor_rng(cfg["seed"], g.name, t.name)
        shp = t.shape
        if len(shp) == 0:
            out[t.name] = [[r.next_float()]]
        elif len(shp) == 1:
            out[t.name] = r.vector(shp[0])
        else:
            out[t.name] = r.matrix(shp[0], shp[-1])
    return out


def shapes_of(g: Graph) -> dict[str, tuple[int, ...]]:
    return {t.name: t.shape for t in g.tensors}


def base_environment(g: Graph, cfg: dict[str, Any]) -> dict[str, list[list[float]]]:
    """在**原图**上跑一遍真值，拿到**每一个张量**的值（含全部中间结果）。

    ⚠️ 这一步不是为了「顺便算个真值」，它是 `const_fold` 能成立的前提：
    `const_fold` 把 `c = b1 + b2` 折叠成**一个 const 叶子**，
    而叶子没有生产者 —— 它的值只能从**被折叠掉的那个表达式的真值**来。

    不这么做会有两种症状，都不像「常量没值」：
    - 直接崩：`KeyError: 'c'`（阶梯里 execute 找不到这个张量）；
    - 更坏的一种：拿 `synth_feeds` 给 `c` 合成一个**随机**常量 ⇒
      程序能跑完、数字也「正常」，只是**折叠后的输出与真值不等** ——
      于是报告里出现「语义核对红了」，而看起来像 fusion/tiling 的锅。
    """
    return eval_graph(g, synth_feeds(g, cfg))


def feeds_for(g: Graph, env: dict[str, list[list[float]]]) -> dict[str, list[list[float]]]:
    """某一级图的输入 = 它的**叶子张量**在原图真值环境里的值。"""
    return {name: env[name] for name in g.inputs()}


# --------------------------------------------------------------------- 三层核对
def check_one(g: Graph, strat: dict[str, Any], cfg: dict[str, Any],
              feeds: dict[str, list[list[float]]],
              ref: dict[str, list[list[float]]],
              shapes: dict[str, tuple[int, ...]],
              program: Any,
              out_names: tuple[str, ...]) -> dict[str, Any]:
    """对一个 `(图, 策略)` 做**语义**与**双成本**两条核对。

    ⚠️ `ref` 与 `out_names` 都来自**原图**（真值），不是这一级图 ——
    图级 pass 会改输出名（`normalize` 可能把输出改成上游张量），
    拿「这一级图的输出名」去真值里找，会找不到或找到别的东西。
    """
    res: ExecResult = execute(program, feeds, shapes)

    drift = 0.0
    ok = True
    for name in out_names:
        if not bitwise_equal(ref[name], res.env[name]):
            ok = False
        drift = max(drift, max_abs_diff(ref[name], res.env[name]))

    static_g = program.global_elems()
    dyn_g = res.global_elems()
    static_s = program.shared_elems()
    dyn_s = res.shared_elems()

    return {
        "graph": g.name,
        "strategy_id": strategy_id(strat),
        "semantic_bitwise_equal": ok,
        "max_abs_drift": drift,
        "static_global_elems": static_g,
        "dynamic_global_elems": dyn_g,
        "global_agrees": static_g == dyn_g,
        "static_shared_elems": static_s,
        "dynamic_shared_elems": dyn_s,
        "shared_agrees": static_s == dyn_s,
        "cost_agrees": static_g == dyn_g and static_s == dyn_s,
    }


def check_all(graphs: tuple[Graph, ...], cfg: dict[str, Any]) -> dict[str, Any]:
    """全部 `(图, 策略)` 的语义 + 双成本核对。

    ⚠️ 核对的是**最终那一级**的程序 —— 它才是「编译器交出去的核」。
    中间级由 `build_ladder` 负责（每一级都要能跑、都要计数一致）。
    """
    rows: list[dict[str, Any]] = []
    for g in graphs:
        env = base_environment(g, cfg)
        ref = {n: env[n] for n in g.outputs}
        for strat in enumerate_strategies(cfg):
            stages = lower_ladder(g, strat, cfg)
            last = stages[-1]
            rows.append(check_one(g, strat, cfg,
                                  feeds_for(last.lowering.graph, env), ref,
                                  shapes_of(last.lowering.graph),
                                  last.program, g.outputs))
    n = len(rows)
    return {
        "n_checked": n,
        "n_semantic_ok": sum(1 for r in rows if r["semantic_bitwise_equal"]),
        "max_abs_drift": max(r["max_abs_drift"] for r in rows) if rows else 0.0,
        "n_cost_agree": sum(1 for r in rows if r["cost_agrees"]),
        "n_global_agree": sum(1 for r in rows if r["global_agrees"]),
        "n_shared_agree": sum(1 for r in rows if r["shared_agrees"]),
        "rows": rows,
    }


# --------------------------------------------------------------------- 阶梯
def _ladder_for(graphs: tuple[Graph, ...], strat: dict[str, Any],
                cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """一条降级链 × 每张图。每一级都**真的跑一遍**并给出**动态**计数。"""
    per: dict[str, list[dict[str, Any]]] = {}
    for g in graphs:
        env = base_environment(g, cfg)
        stages = lower_ladder(g, strat, cfg)
        rows = []
        for st in stages:
            sg = st.lowering.graph
            feeds = feeds_for(sg, env)
            shapes = shapes_of(sg)
            rd = C.program_cost(st.program, cfg["cost"])
            ex = execute(st.program, feeds, shapes)
            row = st.to_dict()
            row["readings"] = {k: rd[k] for k in C.READING_KEYS}
            row["dynamic_global_elems"] = ex.global_elems()
            row["dynamic_shared_elems"] = ex.shared_elems()
            # ⚠️ **每一级的核也要进产物**。这个项目的卖点之一是
            #    「每一级 IR 都是产物」，只留聚合读数的话，那句话是假的：
            #    「每个核都至少写一次全局内存」「含行归约的核列 tile 盖满整行」
            #    这类断言就只能靠「build 层应该拦住了吧」来交代 ——
            #    而**靠别人拦住**的断言，在别人没拦住时不会变红。
            row["kernels"] = [k.to_dict() for k in st.program.kernels]
            rows.append(row)
        per[g.name] = rows
    return per


def build_ladder(graphs: tuple[Graph, ...], cfg: dict[str, Any]) -> dict[str, Any]:
    """**两条**降级链：参考链（全开）+ 基线链（什么都不开）。

    ⚠️ 为什么必须有第二条：参考链是全开的，**一条 `skipped` 级都没有** ——
    于是「被跳过的 pass 那一级与上一级读数完全相同」这条断言在参考链上
    **一次都没执行过**。它照样是绿的，因为它占着一个「这里验过了」的位置，
    而实际上什么都没验。恒真的判据比没有判据更危险。

    ⚠️ `skipped` 这个标志说的是「**这个 pass 有没有被跑**」，不是
    「它有没有改变东西」：`tiling = "none"` 时 tiling pass **跑了**、
    决定不分块 ⇒ `skipped = False`。只有布尔轴取 `False` 才是真的没跑。
    两者混起来的话，「跳过级读数不变」会被误读成「不分块那一级也该不变」。

    基线链同时是一个有用的对照物：它回答「编译器到底做了多少事」——
    基线链的读数就是「什么都不做」的那一端。
    """
    full = CMP.full_strategy(cfg)
    base = CMP.baseline_strategy(cfg)
    per = _ladder_for(graphs, full, cfg)
    per_base = _ladder_for(graphs, base, cfg)
    return {
        "strategy": {k: full[k] for k in sorted(full)},
        "n_stages": len(next(iter(per.values()))) if per else 0,
        "per_graph": per,
        #: 基线链：布尔轴全 `False`，枚举轴取各自的**第一项**（最省的那一档）。
        "baseline_strategy": {k: base[k] for k in sorted(base)},
        "baseline": per_base,
    }


# --------------------------------------------------------------------- 主入口
def run_pipeline(cfg: dict[str, Any], out_dir: str | Path, *,
                 write: bool = True) -> Run:
    t0 = time.perf_counter()
    out = Path(out_dir)
    graphs = load_graphs(cfg)
    t_load = time.perf_counter()

    ladder = build_ladder(graphs, cfg)
    t_ladder = time.perf_counter()

    comparison = CMP.compare_all(graphs, cfg)
    t_compare = time.perf_counter()

    attribution = AT.attribute_all(graphs, cfg)
    t_attr = time.perf_counter()

    checks = check_all(graphs, cfg)
    t_check = time.perf_counter()

    products: dict[str, Any] = {
        "graphs.json": {
            "n_graphs": len(graphs),
            "graphs": [g.to_dict() for g in graphs],
        },
        "ladder.json": ladder,
        "compare.json": comparison,
        "attribution.json": attribution,
        "semantics.json": {
            "n_checked": checks["n_checked"],
            "n_semantic_ok": checks["n_semantic_ok"],
            "max_abs_drift": checks["max_abs_drift"],
            "n_graphs": len(graphs),
            "rows": [{"graph": r["graph"], "strategy_id": r["strategy_id"],
                      "bitwise_equal": r["semantic_bitwise_equal"],
                      "max_abs_drift": r["max_abs_drift"]} for r in checks["rows"]],
        },
        "crosscheck.json": {
            "n_checked": checks["n_checked"],
            "n_cost_agree": checks["n_cost_agree"],
            "n_global_agree": checks["n_global_agree"],
            "n_shared_agree": checks["n_shared_agree"],
            "rows": [{"graph": r["graph"], "strategy_id": r["strategy_id"],
                      "static_global_elems": r["static_global_elems"],
                      "dynamic_global_elems": r["dynamic_global_elems"],
                      "static_shared_elems": r["static_shared_elems"],
                      "dynamic_shared_elems": r["dynamic_shared_elems"],
                      "agrees": r["cost_agrees"]} for r in checks["rows"]],
        },
    }

    # ---- 三层核对 ----
    # ⚠️ 顺序**不是随意的**，而且里面有一个**环**要拆：
    #
    #   报告要印对抗用例的结论 ⇒ `render` 要读 `summary.json` 与 `adversarial.json`
    #   而 `adversarial.json` 正是这一层在算的东西。
    #
    # 拆法：对抗用例跑**两趟**（`invariants.REPORT_DEPENDENT_CASES`）——
    # 第一趟判「产物的性质」，它的结论让 `summary.json` 与报告都能成型；
    # 第二趟判「渲染器自己的性质」，那时它才真的可以渲染。
    #
    # 不拆的后果不是报错，是**两个数字对不上**：
    # `render` 抛 `KeyError: 'summary.json'` 被 `except Exception` 兜成「违反」
    # ⇒ `run` 说 47/48、`check` 说 48/48。**说的同一件事，却对不上。**
    products["acceptance.json"] = INV.run_acceptance(products, cfg)
    products["expectations.json"] = EXP.evaluate(products, cfg)

    products["config.json"] = {
        "protocol_digest": protocol_digest(cfg),
        "resolved": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    products["benchmark.json"] = {
        "timings": {
            "load_graphs": round(t_load - t0, 9),
            "ladder": round(t_ladder - t_load, 9),
            "compare": round(t_compare - t_ladder, 9),
            "attribution": round(t_attr - t_compare, 9),
            "semantics_and_cost": round(t_check - t_attr, 9),
            "acceptance_expectations_summary": round(
                time.perf_counter() - t_check, 9),
        },
    }
    # ⚠️ 版本信封的**声明部分**先放进去（还没有摘要值和时间戳）。
    #    理由不是「顺手」：`no_elapsed_time_key_in_the_versioned_products`
    #    靠 `versioned_files` 决定要走哪些文件，声明缺席时它会**恒绿**。
    #    `version` / `created_at` 在写盘那一步才填 —— 它们本来就要等到
    #    摘要算完（自指）。
    products["version.json"] = {
        "version": "(pending)",
        "protocol_digest": products["config.json"]["protocol_digest"],
        "created_at": None,
        "versioned_files": list(E.VERSIONED_FILES),
        "excluded": E.EXCLUDED,
    }

    missing = [k for k in ADVERSARIAL_INPUTS if k not in products]
    if missing:
        raise AssertionError(
            f"跑对抗用例之前缺少这些产物：{missing} —— "
            "它们在用例里会被读到，缺了会被 `except Exception` 兜成「违反」，"
            "而同一批用例在 `check` 里读盘上的完整产物又是绿的"
        )

    # ---- 第一趟：产物的性质 ----
    products["adversarial.json"] = INV.run_adversarial(
        products, cfg, graphs, exclude=INV.REPORT_DEPENDENT_CASES)
    products["summary.json"] = _summary(products, cfg, graphs)
    products["benchmark.json"]["timings"]["total"] = round(
        time.perf_counter() - t0, 9)

    # ---- 第二趟：渲染器自己的性质（现在才判得了）----
    late = INV.run_adversarial(products, cfg, graphs,
                               only=INV.REPORT_DEPENDENT_CASES)
    products["adversarial.json"] = INV.merge_adversarial(
        products["adversarial.json"], late)
    # 第一趟的计数不含第二趟 ⇒ 必须重算，否则 `summary.json` 里的
    # 「对抗用例 47/48」与 `adversarial.json` 里的 48 条对不上。
    products["summary.json"] = _summary(products, cfg, graphs)
    t_checks = time.perf_counter()

    if write:
        out.mkdir(parents=True, exist_ok=True)
        # ⚠️ `version.json` 要在**写盘前**定稿：它是自指的，而且 `summary.json`
        #    与报告里都引用了版本号 ⇒ 「先写别的、再补版本号」会让盘上出现
        #    一份引用了 `(pending)` 的 summary。
        version = E.digest_of(products)
        products["version.json"] = {
            **products["version.json"],
            "version": f"v1-{version}",
            "created_at": _utc_now(),
        }
        for name, obj in products.items():
            E.write_product(out, name, obj)
        from . import report as R
        md, html = R.render(products)
        # ⚠️ `newline="\n"` 的理由同 `emit.write_product`（C.345）：
        #    不加的话盘上文件是 CRLF，字节数与渲染文本对不上。
        (out / "report.md").write_text(md, encoding="utf-8", newline="\n")
        (out / "report.html").write_text(html, encoding="utf-8", newline="\n")

    return Run(products=products, graphs=graphs, out_dir=out)


def _utc_now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _summary(products: dict[str, Any], cfg: dict[str, Any],
             graphs: tuple[Graph, ...]) -> dict[str, Any]:
    """头条数字。

    ⚠️ 它在对抗用例的**两趟之间**被调一次、**两趟之后**再被调一次 ——
    因为第一趟结束时 `adversarial.json` 已经有了（缺第二趟那两条），
    而报告要印对抗用例的结论。不重算的后果是 `summary.json` 说 47/48、
    `adversarial.json` 里有 48 条 —— **两个数字说的是同一件事，却对不上**。
    """
    acc = products["acceptance.json"]
    adv = products["adversarial.json"]
    exp = products["expectations.json"]
    cmp_ = products["compare.json"]
    sem = products["semantics.json"]
    cc = products["crosscheck.json"]
    return {
        "n_graphs": len(graphs),
        "n_strategies": cmp_["n_strategies"],
        "n_cells": cmp_["matrix_expected_total"],
        "matrix_gap": cmp_["matrix_gap_total"],
        "n_stages": products["ladder.json"]["n_stages"],
        "n_passes": len(cfg["passes"]["order"]),
        "n_optional_passes": len(cfg["passes"]["optional"]),
        "acceptance": {"n": acc["n"], "n_passed": acc["n_passed"],
                       "n_failed": acc["n_failed"]},
        "adversarial": {"n_cases": adv["n_cases"], "n_held": adv["n_held"],
                        "n_violated": adv["n_violated"]},
        "expectations": {"n": exp["n"], "n_held": exp["n_held"],
                         "n_refuted": exp["n_refuted"],
                         "n_unmeasurable": exp["n_unmeasurable"]},
        "semantics": {"n_checked": sem["n_checked"],
                      "n_bitwise_equal": sem["n_semantic_ok"],
                      "max_abs_drift": sem["max_abs_drift"]},
        "crosscheck": {"n_checked": cc["n_checked"], "n_agree": cc["n_cost_agree"]},
        "n_pareto_total": sum(p["n_pareto"] for p in cmp_["per_graph"]),
        "protocol_digest": products["config.json"]["protocol_digest"],
    }
