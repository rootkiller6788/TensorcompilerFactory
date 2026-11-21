"""两层核对：**门槛**（可调达标线）+ **对抗用例**（不可调的不变量断言）。

⚠️ 两层**不能混成一个词**：
- 门槛写在 `config.acceptance` 里，**可以调**，调它就是承认「这条线我放松了」；
- 对抗用例**没有阈值**，它们是「换个输入也必须成立」的断言，调不了。

一条对抗用例如果只是把门槛抄一遍，那它不是不变量，是**门槛的影子** ——
影子会跟着门槛一起放松，于是「放松门槛」看起来什么代价都没有。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import cost as C
from .ir import Graph

# ===================================================================== 门槛
@dataclass(frozen=True)
class Threshold:
    key: str
    label: str
    fn: Callable[[dict[str, Any], dict[str, Any]], tuple[float, bool]]
    why: str = ""


def _min_over_graphs(ladder: dict[str, Any], f: Callable[[list[dict]], float]) -> float:
    vals = [f(rows) for rows in ladder["per_graph"].values()]
    return min(vals) if vals else 0.0


def _stage(rows: list[dict], name: str) -> dict | None:
    for r in rows:
        if r["name"] == name:
            return r
    return None


def _bytes_at(rows: list[dict], name: str | None) -> float:
    if name is None:
        return float(rows[-1]["readings"]["global_bytes"])
    st = _stage(rows, name)
    if st is None:
        raise KeyError(f"降级链里没有名为 {name!r} 的一级 —— "
                       "门槛引用了不存在的一级，它永远是绿的")
    return float(st["readings"]["global_bytes"])


def _tiling_gain_over_fusion(rows: list[dict]) -> float:
    """分块相对**融合之后**的额外访存增益。"""
    after_fusion = _bytes_at(rows, "fusion")
    after_tiling = _bytes_at(rows, "tiling")
    return after_fusion / after_tiling if after_tiling else float("inf")


def _fusion_kernel_reduction(rows: list[dict]) -> float:
    """融合把**核数**降了几倍（`as_written` → `fusion` 那一级）。"""
    after = _bytes_at_kernels(rows, "fusion")
    naive = float(rows[0]["n_kernels"])
    return naive / after if after else float("inf")


def _bytes_at_kernels(rows: list[dict], name: str) -> float:
    st = _stage(rows, name)
    if st is None:
        raise KeyError(f"降级链里没有名为 {name!r} 的一级 —— "
                       "门槛引用了不存在的一级，它永远是绿的")
    return float(st["n_kernels"])


def _total_reduction(rows: list[dict]) -> float:
    naive = float(rows[0]["readings"]["global_bytes"])
    final = float(rows[-1]["readings"]["global_bytes"])
    return naive / final if final else float("inf")


def _min_kernels(rows: list[dict]) -> float:
    return float(min(r["n_kernels"] for r in rows))


def _min_distinct_costs(products: dict[str, Any]) -> float:
    out = []
    for p in products["compare.json"]["per_graph"]:
        keys = {tuple(sorted((k, v) for k, v in c["readings"].items()))
                for c in p["cells"]}
        out.append(float(len(keys)))
    return min(out) if out else 0.0


def _graph_pass_count(products: dict[str, Any], meta_key: str) -> float:
    """某个图级 pass 在**至少一张图**上省下的算子数（**取最大**）。

    ⚠️ 取 `max` 不取 `min`。这三条门槛问的是「**这一级有没有活干**」，
    不是「每一张图都必须有活干」。取 `min` 会让 `dual` 的重复子表达式
    被 `attn`（本来就没有重复）抵消成 0 —— 于是「CSE 这一级有没有被验到」
    这个问题被答成了「CSE 在 attn 上有没有用」，而后者本来就该是 0。
    ⇒ 一个「每张图都必须有重复子表达式」的门槛，是在要求数据长得像
      为了门槛而造的数据，不是在验编译器。
    """
    out = []
    for rows in products["ladder.json"]["per_graph"].values():
        for r in rows:
            if meta_key in r["meta"]:
                out.append(float(r["meta"][meta_key]))
    return max(out) if out else 0.0


ACCEPTANCE_CHECKS: tuple[Threshold, ...] = (
    Threshold("min_graphs", "图的数量 ≥ {v}",
              lambda p, c: (len(p["graphs.json"]["graphs"]),
                            len(p["graphs.json"]["graphs"]) >= c["acceptance"]["min_graphs"]),
              "图太少 ⇒ 结论是「这一个图恰好如此」"),
    Threshold("min_passes", "pass 数 ≥ {v}",
              lambda p, c: (len(c["passes"]["order"]),
                            len(c["passes"]["order"]) >= c["acceptance"]["min_passes"]),
              "降级链太短 ⇒ 漏斗只有两格，谈不上「逐级」"),
    Threshold("min_strategies", "策略数 ≥ {v}",
              lambda p, c: (p["compare.json"]["n_strategies"],
                            p["compare.json"]["n_strategies"] >= c["acceptance"]["min_strategies"]),
              "策略太少 ⇒ 「多策略对比」没有信息量"),
    Threshold("min_funnel_stages", "每图降级级数 ≥ {v}",
              lambda p, c: (p["ladder.json"]["n_stages"],
                            p["ladder.json"]["n_stages"] >= c["acceptance"]["min_funnel_stages"]),
              "级数不够 ⇒ 漏斗的刻度太粗，归因无从下手"),
    Threshold("min_kernels_per_graph", "每图最少核数 ≥ {v}",
              lambda p, c: (_min_over_graphs(p["ladder.json"], _min_kernels),
                            _min_over_graphs(p["ladder.json"], _min_kernels)
                            >= c["acceptance"]["min_kernels_per_graph"]),
              "0 个核 ⇒ 这张图什么都没编出来"),
    Threshold("min_fusion_kernel_reduction", "融合的核数降幅 ≥ {v}×",
              lambda p, c: (_min_over_graphs(p["ladder.json"], _fusion_kernel_reduction),
                            _min_over_graphs(p["ladder.json"], _fusion_kernel_reduction)
                            >= c["acceptance"]["min_fusion_kernel_reduction"]),
              "融合不省核数 ⇒ 这个 pass 是装饰。"
              "⚠️ 门槛盯的是**核数**不是访存：不分块时矩阵乘的访存是 O(MNK)，"
              "融合省掉的是 O(MN) 那一小截，比例可以低到 1.01 —— "
              "拿访存当融合的达标线，会把一个真的有杀伤的 pass 判成装饰"),
    Threshold("min_tiling_gain_over_fusion", "分块相对融合的额外增益 ≥ {v}×",
              lambda p, c: (_min_over_graphs(p["ladder.json"], _tiling_gain_over_fusion),
                            _min_over_graphs(p["ladder.json"], _tiling_gain_over_fusion)
                            >= c["acceptance"]["min_tiling_gain_over_fusion"]),
              "分块没有额外增益 ⇒ 矩阵乘的 O(MNK) 访存没被修掉"),
    Threshold("min_total_traffic_reduction", "整条链的总访存降幅 ≥ {v}×",
              lambda p, c: (_min_over_graphs(p["ladder.json"], _total_reduction),
                            _min_over_graphs(p["ladder.json"], _total_reduction)
                            >= c["acceptance"]["min_total_traffic_reduction"]),
              "总降幅太小 ⇒ 「编译器」这个名字撑不住"),
    Threshold("min_attributed_passes", "至少 {v} 个 pass 有杀伤",
              lambda p, c: (p["attribution.json"]["n_load_bearing_some"],
                            p["attribution.json"]["n_load_bearing_some"]
                            >= c["acceptance"]["min_attributed_passes"]),
              "没有 pass 有杀伤 ⇒ 整条链是摆设"),
    Threshold("min_lowering_paths_with_distinct_cost", "每图的不同成本向量 ≥ {v}",
              lambda p, c: (_min_distinct_costs(p),
                            _min_distinct_costs(p)
                            >= c["acceptance"]["min_lowering_paths_with_distinct_cost"]),
              "所有策略成本相同 ⇒ 「多策略」是假的"),
    Threshold("min_cost_crosscheck_agreement", "静态/动态成本一致率 ≥ {v}",
              lambda p, c: (_rate(p, "crosscheck.json", "n_cost_agree", "n_checked"),
                            _rate(p, "crosscheck.json", "n_cost_agree", "n_checked")
                            >= c["acceptance"]["min_cost_crosscheck_agreement"]),
              "两条独立路径算同一个量却不相等 ⇒ 至少有一条错了"),
    Threshold("max_semantic_drift", "语义漂移 ≤ {v}",
              lambda p, c: (p["semantics.json"]["max_abs_drift"],
                            p["semantics.json"]["max_abs_drift"]
                            <= c["acceptance"]["max_semantic_drift"]),
              "降级改变了数值 ⇒ 编译器编错了"),
    Threshold("min_cse_saved_ops", "CSE 至少省 {v} 个算子",
              lambda p, c: (_graph_pass_count(p, "n_merged"),
                            _graph_pass_count(p, "n_merged")
                            >= c["acceptance"]["min_cse_saved_ops"]),
              "图上没有可消的重复 ⇒ CSE 这一级是空的（要换图，不是调门槛）"),
    Threshold("min_const_fold_saved_ops", "常量折叠至少省 {v} 个算子",
              lambda p, c: (_graph_pass_count(p, "n_folded"),
                            _graph_pass_count(p, "n_folded")
                            >= c["acceptance"]["min_const_fold_saved_ops"]),
              "没有常量子图 ⇒ 这一级是空的"),
    Threshold("min_normalize_saved_ops", "代数简化至少省 {v} 个算子",
              lambda p, c: (_graph_pass_count(p, "n_saved_ops"),
                            _graph_pass_count(p, "n_saved_ops")
                            >= c["acceptance"]["min_normalize_saved_ops"]),
              "没有恒等算子 ⇒ 这一级是空的"),
)


def _rate(p: dict[str, Any], product: str, num: str, den: str) -> float:
    d = p[product][den]
    return (p[product][num] / d) if d else 1.0


def run_acceptance(products: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for t in ACCEPTANCE_CHECKS:
        value, ok = t.fn(products, cfg)
        rows.append({
            "key": t.key,
            "label": t.label.format(v=cfg["acceptance"][t.key]),
            "threshold": cfg["acceptance"][t.key],
            "observed": value,
            "passed": bool(ok),
            "why": t.why,
        })
    return {
        "n": len(rows),
        "n_passed": sum(1 for r in rows if r["passed"]),
        "n_failed": sum(1 for r in rows if not r["passed"]),
        "rows": rows,
    }


# ===================================================================== 对抗用例
@dataclass(frozen=True)
class Case:
    name: str
    claim: str
    fn: Callable[[dict[str, Any], dict[str, Any], tuple[Graph, ...]], tuple[bool, str]]


def _case(name: str, claim: str):
    def deco(fn):
        return Case(name, claim, fn)
    return deco


def _ladder_rows(products: dict[str, Any]) -> dict[str, list[dict]]:
    return products["ladder.json"]["per_graph"]


# ---- 语义 / 成本 ------------------------------------------------------
@_case("semantics_is_bit_exact_everywhere",
       "每一张图 × 每一条策略：降级后的输出与真值**逐位相等**")
def _c_sem(p, c, gs):
    s = p["semantics.json"]
    bad = [r for r in s["rows"] if not r["bitwise_equal"]]
    return (not bad, f"{s['n_semantic_ok']}/{s['n_checked']} 逐位相等；"
                     f"最大漂移 {s['max_abs_drift']:.3e}；不等 {len(bad)} 条")


@_case("cost_crosscheck_everywhere",
       "静态解析式与动态计数在**全部** (图, 策略) 上逐位相等")
def _c_cc(p, c, gs):
    x = p["crosscheck.json"]
    bad = [r for r in x["rows"] if not r["agrees"]]
    return (not bad, f"{x['n_cost_agree']}/{x['n_checked']} 一致；"
                     f"不一致 {len(bad)} 条")


@_case("matrix_gap_is_zero",
       "对比矩阵没有空格（缺口 = 声明格数 − 实填格数，**算出来的**）")
def _c_gap(p, c, gs):
    g = p["compare.json"]["matrix_gap_total"]
    return (g == 0, f"缺口 {g}（声明 {p['compare.json']['matrix_expected_total']} 格）")


# ---- 漏斗 ------------------------------------------------------------
@_case("funnel_is_monotone_in_global_bytes",
       "沿 pass 链，全局访存**逐级不增**（没有哪个 pass 让程序做更多全局访存）")
def _c_mono(p, c, gs):
    bad = []
    for gname, rows in _ladder_rows(p).items():
        prev = None
        for r in rows:
            v = r["readings"]["global_bytes"]
            if prev is not None and v > prev:
                bad.append(f"{gname}@{r['name']}: {prev} → {v}")
            prev = v
    return (not bad, "全部单调不增" if not bad else "上升：" + "; ".join(bad))


@_case("funnel_is_monotone_in_kernel_count",
       "沿 pass 链，核数**逐级不增**")
def _c_mono_k(p, c, gs):
    bad = []
    for gname, rows in _ladder_rows(p).items():
        prev = None
        for r in rows:
            v = r["n_kernels"]
            if prev is not None and v > prev:
                bad.append(f"{gname}@{r['name']}: {prev} → {v}")
            prev = v
    return (not bad, "全部单调不增" if not bad else "上升：" + "; ".join(bad))


@_case("ladder_covers_every_pass",
       "降级链的级数 = pass 数 + 1（起点 `as_written`），且名字逐条对上")
def _c_ladder_names(p, c, gs):
    want = ["as_written"] + list(c["passes"]["order"])
    bad = []
    for gname, rows in _ladder_rows(p).items():
        got = [r["name"] for r in rows]
        if got != want:
            bad.append(f"{gname}: {got}")
    return (not bad, "每张图的级名与 passes.order 逐条一致"
                     if not bad else "; ".join(bad))


@_case("every_optional_pass_appears_in_the_ladder",
       "每一个可选 pass 都在降级链里出现过（否则它永远不会被跑到）")
def _c_optional_present(p, c, gs):
    opt = set(c["passes"]["optional"])
    seen = set()
    for rows in _ladder_rows(p).values():
        seen |= {r["name"] for r in rows}
    miss = sorted(opt - seen)
    return (not miss, "全部出现" if not miss else f"缺失 {miss}")


@_case("skipped_stages_do_not_change_the_program",
       "被跳过的 pass 那一级与上一级**读数完全相同**（跳过就是没做）")
def _c_skip(p, c, gs):
    """⚠️ 这条**必须读两条链**。

    参考链是「可选 pass 全开」的 ⇒ 里面**一条 `skipped` 级都没有**。
    只读参考链的话，这个循环体一次都不执行，`bad` 恒为空 ——
    它照样返回「跳过级读数不变」，因为它占着一个「这里验过了」的位置，
    而实际上什么都没验。

    ⇒ 同时读 `baseline`（什么都不开）那条链，并且要求
    **真的检查到至少一个 `skipped` 级**（`n_skipped > 0`）——
    否则这条用例应该红，而不是绿着假装验过。
    """
    lad = p["ladder.json"]
    bad = []
    n_skipped = 0
    for key in ("per_graph", "baseline"):
        for gname, rows in lad.get(key, {}).items():
            for i, r in enumerate(rows):
                if r["skipped"] and i > 0:
                    n_skipped += 1
                    if rows[i - 1]["readings"] != r["readings"]:
                        bad.append(f"{key}/{gname}@{r['name']}")
    return (not bad and n_skipped > 0,
            f"{n_skipped} 个跳过级，读数都与上一级相同" if not bad and n_skipped
            else ("产物里一条 `skipped` 级都没有 —— 这条用例空转" if not n_skipped
                  else "跳过却变了：" + "; ".join(bad)))


# ---- 核的结构 --------------------------------------------------------
@_case("launch_count_equals_kernel_count",
       "kernel launch 数 == 核数（不存在一个核多次 launch）")
def _c_launch(p, c, gs):
    bad = []
    for gname, rows in _ladder_rows(p).items():
        for r in rows:
            if r["readings"]["n_kernels"] != r["n_kernels"]:
                bad.append(f"{gname}@{r['name']}")
    return (not bad, "每级都相等" if not bad else "; ".join(bad))


@_case("every_kernel_stores_something",
       "每个核都至少有一次全局写（只读不写的核是死代码）")
def _c_store(p, c, gs):
    bad = []
    n_seen = 0
    for gname, rows in _ladder_rows(p).items():
        for r in rows:
            for k in r.get("kernels", []):
                n_seen += 1
                store = sum(m["elems"] for m in k["mem"]
                            if m["space"] == "global" and m["kind"] == "store")
                if store <= 0:
                    bad.append(f"{gname}@{r['name']}/{k['kid']}")
    return (not bad and n_seen > 0,
            f"{n_seen} 个核，每个都有全局写" if not bad and n_seen
            else (f"没读到任何核（产物里没有 kernels 字段？）" if not n_seen
                  else f"没有写的核：{bad[:3]}"))


@_case("partition_covers_every_node_exactly_once",
       "分块（partition）把**这一级图**的每个算子恰好放进一个核，不重不漏")
def _c_partition(p, c, gs):
    """⚠️ 比对的是**这一格最终的图**，不是 `data/graphs/*.json` 里那张原图。

    图级 pass 会删算子（normalize / const_fold / cse），拿原图的算子名去比对，
    会在每一个「图级 pass 生效」的格子上报不重不漏失败 —— 而那 160 格完全正确。
    ⇒ 判据必须与声明指同一个对象：「分块覆盖了这一级图的全部算子」。
    """
    bad = []
    n_cells = 0
    for pg in p["compare.json"]["per_graph"]:
        for cell in pg["cells"]:
            n_cells += 1
            want = list(cell["final_nodes"])
            got = sorted(x for grp in cell["partition"] for x in grp)
            if got != want:
                bad.append(f"{pg['graph']}@{cell['strategy_id']}")
    return (not bad, f"{n_cells} 格都不重不漏" if not bad
                     else f"{len(bad)} 格不对：{bad[:3]}")


@_case("fusing_never_increases_kernel_count",
       "融合后的核数 ≤ 融合前")
def _c_fuse_k(p, c, gs):
    bad = []
    for gname, rows in _ladder_rows(p).items():
        f = _stage(rows, "fusion")
        if f is None:
            continue
        idx = rows.index(f)
        if f["n_kernels"] > rows[idx - 1]["n_kernels"]:
            bad.append(gname)
    return (not bad, "全部不增" if not bad else f"{bad}")


@_case("cse_never_increases_op_count",
       "CSE 后的算子数 ≤ CSE 前")
def _c_cse(p, c, gs):
    bad = []
    for gname, rows in _ladder_rows(p).items():
        s = _stage(rows, "cse")
        if s is None:
            continue
        idx = rows.index(s)
        if s["n_ops"] > rows[idx - 1]["n_ops"]:
            bad.append(gname)
    return (not bad, "全部不增" if not bad else f"{bad}")


@_case("const_fold_never_increases_op_count",
       "常量折叠后的算子数 ≤ 折叠前")
def _c_cf(p, c, gs):
    bad = []
    for gname, rows in _ladder_rows(p).items():
        s = _stage(rows, "const_fold")
        if s is None:
            continue
        idx = rows.index(s)
        if s["n_ops"] > rows[idx - 1]["n_ops"]:
            bad.append(gname)
    return (not bad, "全部不增" if not bad else f"{bad}")


@_case("normalize_never_increases_op_count",
       "代数简化后的算子数 ≤ 简化前")
def _c_norm(p, c, gs):
    bad = []
    for gname, rows in _ladder_rows(p).items():
        s = _stage(rows, "normalize")
        if s is None:
            continue
        idx = rows.index(s)
        if s["n_ops"] > rows[idx - 1]["n_ops"]:
            bad.append(gname)
    return (not bad, "全部不增" if not bad else f"{bad}")


@_case("flops_never_increase_along_the_ladder",
       "沿 pass 链，工作量读数**逐级不增**（图级 pass 删冗余会让它降，程序级 pass 一步都不改它）")
def _c_flops(p, c, gs):
    """⚠️ 这条原来是「沿整条链**恒定**」，而那是**错的**：

    `cse` 把 `dual` 里算了两遍的同一个 matmul 合并成一个 ⇒ 工作量**真的减半**；
    `normalize` / `const_fold` 删掉算子 ⇒ 工作量也跟着降。
    恒定的说法只有在「flops 只数矩阵乘、且图级 pass 不动矩阵乘」时才成立，
    而 CSE 恰恰会动矩阵乘。

    ⇒ 改成一个**方向性**的断言（不增），并把它拆成两条：
    这里管整条链的方向，`program_passes_do_not_change_flops` 管
    「程序级 pass 一步都不改」——后者才是「改怎么算 / 改算什么」那条分界线。
    """
    bad = []
    for gname, rows in _ladder_rows(p).items():
        prev = None
        for r in rows:
            v = r["readings"]["flops"]
            if prev is not None and v > prev:
                bad.append(f"{gname}@{r['name']}: {prev} → {v}")
            prev = v
    return (not bad, "全部单调不增" if not bad else "上升：" + "; ".join(bad))


@_case("program_passes_do_not_change_flops",
       "fusion/tiling/layout/memory_plan/lower **一步都不改工作量**"
       "（它们只改「怎么算」，不改「算什么」）")
def _c_flops_prog(p, c, gs):
    """程序级 pass 的**定义**就是「不改算了多少」。

    ⚠️ 这条不能用「总工作量不增」代替：不增是**弱**断言，
    融合时多算一点（比如为了对齐而重算）也能满足它。
    这里逐级比**相等**，把「改怎么算」和「改算什么」真正分开。
    """
    prog = ("fusion", "tiling", "layout", "memory_plan", "lower")
    bad = []
    n_checked = 0
    for gname, rows in _ladder_rows(p).items():
        for i, stage in enumerate(rows):
            if stage["name"] not in prog or i == 0:
                continue
            n_checked += 1
            if stage["readings"]["flops"] != rows[i - 1]["readings"]["flops"]:
                bad.append(f"{gname}@{stage['name']}: "
                           f"{rows[i - 1]['readings']['flops']} → "
                           f"{stage['readings']['flops']}")
    return (not bad and n_checked > 0,
            f"{n_checked} 级程序级 pass，工作量逐级不变" if not bad and n_checked
            else ("没查到程序级 pass" if not n_checked else f"变了：{bad[:3]}"))


@_case("flops_vary_only_with_the_graph_level_axis",
       "同一张图的全部策略里，工作量**只随 `cse` 变**（其余四根轴一个都不改它）")
def _c_flops_s(p, c, gs):
    """⚠️ 原来写的是「所有策略给出同一个总工作量」，而那是**错的**：
    `cse` 是一根**策略轴**，它关掉时 `dual` 会把同一个 matmul 算两遍。
    ⇒ 正确的说法是「只有图级的那根轴能动工作量」——
    把 `cse` 固定住之后，`fusion/tiling/layout/memory_plan` 怎么变都不改它。
    """
    bad = []
    n_groups = 0
    for pg in p["compare.json"]["per_graph"]:
        by: dict[bool, set[int]] = {}
        for cell in pg["cells"]:
            by.setdefault(bool(cell["strategy"]["cse"]), set()).add(
                cell["readings"]["flops"])
        for flag, vals in sorted(by.items()):
            n_groups += 1
            if len(vals) != 1:
                bad.append(f"{pg['graph']}(cse={flag}): {sorted(vals)}")
    return (not bad, f"{n_groups} 组（图 × cse 开关）内工作量一致" if not bad
                     else "; ".join(bad))


# ---- 成本模型的边界 --------------------------------------------------
@_case("occupancy_never_exceeds_one",
       "占用率 ≤ 1.0（> 1 说明模型算错了）")
def _c_occ(p, c, gs):
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        for cell in pg["cells"]:
            o = cell["readings"]["occupancy"]
            if o > 1.0 + 1e-12 or o < 0.0:
                bad.append(f"{pg['graph']}@{cell['strategy_id']}={o}")
    return (not bad, "全部在 [0,1]" if not bad else f"{bad[:3]}")


@_case("threads_never_exceed_the_hardware_limit",
       "每个核的线程数 ≤ `max_threads_per_sm`")
def _c_threads(p, c, gs):
    lim = c["cost"]["max_threads_per_sm"]
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        for cell in pg["cells"]:
            if cell["readings"]["threads"] > lim:
                bad.append(f"{pg['graph']}@{cell['strategy_id']}")
    return (not bad, f"全部 ≤ {lim}" if not bad else f"{bad[:3]}")


@_case("readings_are_never_negative",
       "所有读数非负（负数比报错更危险 —— 它看起来像个数字）")
def _c_neg(p, c, gs):
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        for cell in pg["cells"]:
            for k, v in cell["readings"].items():
                if isinstance(v, (int, float)) and v < 0:
                    bad.append(f"{pg['graph']}@{cell['strategy_id']}.{k}={v}")
    return (not bad, "全部非负" if not bad else f"{bad[:3]}")


@_case("peak_shared_bytes_fits_the_hardware",
       "峰值 shared 用量 ≤ `shared_per_sm`（超了核根本起不来）")
def _c_shared(p, c, gs):
    lim = c["cost"]["shared_per_sm"]
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        for cell in pg["cells"]:
            if cell["readings"]["peak_shared_bytes"] > lim:
                bad.append(f"{pg['graph']}@{cell['strategy_id']}")
    return (not bad, f"全部 ≤ {lim}" if not bad else f"{bad[:3]}")


@_case("latency_breakdown_sums_to_latency",
       "延迟的四项分解之和 == 总延迟（差一点说明有一项没被算进去）")
def _c_lat_sum(p, c, gs):
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        for cell in pg["cells"]:
            s = sum(cell["latency_breakdown"].values())
            if abs(s - cell["readings"]["latency_us"]) > 1e-9 * max(1.0, abs(s)):
                bad.append(f"{pg['graph']}@{cell['strategy_id']}")
    return (not bad, "全部相等" if not bad else f"{bad[:3]}")


# ---- 前沿 ------------------------------------------------------------
@_case("pareto_points_are_pairwise_non_dominating",
       "前沿上的点两两互不支配")
def _c_pareto(p, c, gs):
    from .compare import dominates
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        pts = pg["pareto"]
        for i, a in enumerate(pts):
            for j, b in enumerate(pts):
                if i != j and dominates(a["readings"], b["readings"]):
                    bad.append(f"{pg['graph']}: {a['strategy_id']} 支配 {b['strategy_id']}")
    return (not bad, "互不支配" if not bad else f"{bad[:3]}")


@_case("best_latency_point_is_on_the_pareto_front",
       "延迟最低的那一格必在前沿上")
def _c_best_on_front(p, c, gs):
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        ids = {x["strategy_id"] for x in pg["pareto"]}
        if pg["best_latency"]["strategy_id"] not in ids:
            bad.append(pg["graph"])
    return (not bad, "都在前沿上" if not bad else f"{bad}")


@_case("best_bytes_is_never_worse_than_baseline",
       "访存最优的策略，访存不差于基线")
def _c_best_vs_base(p, c, gs):
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        if pg["best_bytes"]["readings"]["global_bytes"] > pg["baseline"]["readings"]["global_bytes"]:
            bad.append(pg["graph"])
    return (not bad, "全部不差于基线" if not bad else f"{bad}")


# ---- 归因 ------------------------------------------------------------
@_case("attribution_covers_every_optional_pass",
       "消融覆盖了**每一个**可选 pass（漏掉一个 ⇒ 它的贡献没人知道）")
def _c_attr_cover(p, c, gs):
    opt = set(c["passes"]["optional"])
    bad = []
    for r in p["attribution.json"]["per_graph"]:
        miss = sorted(opt - set(r["per_pass"]))
        if miss:
            bad.append(f"{r['graph']}: 缺 {miss}")
    return (not bad, "全覆盖" if not bad else "; ".join(bad))


@_case("attribution_deltas_are_reported_for_every_reading",
       "每个 pass 的消融都给全部读数一个 delta（不许只报好看的那几个）")
def _c_attr_readings(p, c, gs):
    want = set(C.READING_KEYS)
    bad = []
    for r in p["attribution.json"]["per_graph"]:
        for name, d in r["per_pass"].items():
            miss = want - set(d["delta_without_it"])
            if miss:
                bad.append(f"{r['graph']}/{name}: 缺 {sorted(miss)}")
    return (not bad, "每个 delta 都覆盖全部读数" if not bad else "; ".join(bad))


# ---- 策略空间 --------------------------------------------------------
@_case("strategy_ids_are_unique_per_graph",
       "同一张图里策略 id 唯一（重复 ⇒ 矩阵有格子被覆盖）")
def _c_sid(p, c, gs):
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        ids = [cell["strategy_id"] for cell in pg["cells"]]
        if len(set(ids)) != len(ids):
            bad.append(pg["graph"])
    return (not bad, "唯一" if not bad else f"{bad}")


@_case("matrix_has_exactly_the_declared_number_of_cells",
       "每张图的格子数 == 策略空间的交叉积大小")
def _c_cells(p, c, gs):
    want = 1
    for v in c["strategies"].values():
        want *= len(v)
    bad = [pg["graph"] for pg in p["compare.json"]["per_graph"]
           if len(pg["cells"]) != want]
    return (not bad, f"每图 {want} 格" if not bad else f"{bad}")


@_case("every_strategy_axis_takes_every_declared_value",
       "每根轴上，配置声明的每个取值都真的出现过")
def _c_axis(p, c, gs):
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        for axis, vals in c["strategies"].items():
            seen = {cell["strategy"][axis] for cell in pg["cells"]}
            if seen != set(vals):
                bad.append(f"{pg['graph']}.{axis}: {sorted(seen)}")
    return (not bad, "每根轴都覆盖全部取值" if not bad else f"{bad[:3]}")


# ---- 产物纪律 --------------------------------------------------------
@_case("products_are_json_serializable",
       "全部产物可 JSON 序列化（含 NaN/Inf 的检查）")
def _c_serial(p, c, gs):
    bad = []
    for name, obj in p.items():
        try:
            json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            bad.append(f"{name}: {e}")
    return (not bad, f"{len(p)} 个产物全部可序列化" if not bad else "; ".join(bad))


_TIMING_KEY = re.compile(
    r"(timings?|elapsed|duration|wall_?clock|perf_counter|seconds|_us$)",
    re.IGNORECASE)
_CARDINALITY = re.compile(r"^(n_|num_)|_count$", re.IGNORECASE)
_DECL_PREFIXES = (("config.json", "resolved", "expectations"),
                  ("config.json", "expectations"))

#: **允许进版本摘要的时间类键名** —— 它们是**成本模型的确定性函数**
#: （同一份口径必得同一个值），不是墙钟测量。
#:
#: ⚠️ 这个白名单必须**短**，且每一条写清为什么。它一旦开始变长，
#: 就说明「模型预测」与「实测耗时」在产物里混住了 —— 而那是这个项目
#: 最危险的失真（把假设标成测量）。
#:
#: 白名单是**白名单**不是黑名单：下面 `_c_timing` 收的是**全部**命中
#: 正则的键，然后断言 `命中集合 ⊆ 白名单`。新增一个时间类键名
#: 必须显式来这里报到 —— 否则它是静默漏进去的。
_MODEL_TIME_KEYS = frozenset({
    "latency_us",          # 成本模型算出的延迟 = Σ(launch + 访存 + 算力)
    "launch_overhead_us",  # 成本模型的一个假设常数（每次 launch 3 µs）
})


@_case("no_elapsed_time_key_in_the_versioned_products",
       "**进版本摘要的产物里不许有实测耗时**（否则两次 run 不逐位相同）")
def _c_timing(p, c, gs):
    hits: list[str] = []
    keys: set[str] = set()

    def walk(node, path, under_timing, exempt):
        """`exempt` 沿子树传下去：**遇到白名单里的时间键之后，它的整棵子树都豁免**。

        ⚠️ 必须传标志位，不能在标量那一层回头看 `path[-1]`：
        白名单里的键下面可能是一个**列表**（`"latency_us": [10.6, 9.5, ...]`），
        这时标量层的 `path[-1]` 是 `"[0]"`，回头看不出来源。
        症状是「白名单写了、照样红」，而看起来像白名单没生效。
        """
        if any(path[:len(pre)] == pre for pre in _DECL_PREFIXES):
            return
        if isinstance(node, dict):
            for k in sorted(node):
                key = str(k)
                val = node[k]
                hit = bool(_TIMING_KEY.search(key)) and not _CARDINALITY.search(key)
                if hit:
                    keys.add(key)
                walk(val, path + (key,), under_timing or hit,
                     exempt or (hit and key in _MODEL_TIME_KEYS))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + (f"[{i}]",), under_timing, exempt)
        elif under_timing and not exempt and isinstance(node, (int, float)) \
                and not isinstance(node, bool):
            hits.append(f"{'.'.join(path)}={node!r}")

    for name in p.get("version.json", {}).get("versioned_files", []):
        if name in p:
            walk(p[name], (name,), False, False)
    extra = sorted(keys - _MODEL_TIME_KEYS)
    ok = not hits and not extra
    detail = ("没有实测耗时进摘要；时间类键名恰好是白名单"
              f" {sorted(_MODEL_TIME_KEYS)}" if ok else
              ("命中实测耗时：" + "; ".join(hits[:6]) if hits else
               f"出现了白名单外的时间类键名 {extra} —— "
               "要么它是实测耗时（不该进摘要），要么它该被加进 `_MODEL_TIME_KEYS`"
               "并写清为什么它是确定性的"))
    return (ok, detail)


@_case("adversarial_cases_run_with_all_their_inputs_present",
       "对抗用例是在**产物齐全**的前提下跑的"
       "（否则缺产物会被 `except Exception` 兜成「违反」，而同一批用例读盘上的完整产物又是绿的）")
def _c_adv_inputs(p, c, gs):
    """`run` 与 `check` 对**同一批用例**必须给出同一个裁定。

    ⚠️ 这条是补出来的，因为本轮抓到的真缺陷正是它的反面：
    `adversarial` 曾经跑在 `expectations` 之前，于是
    `expectation_verdicts_are_a_partition` 与
    `renderers_are_pure_functions_of_the_products` 在 `run` 里报「违反」、
    在 `check` 里报「成立」——**两个数字说的是同一件事，却对不上**。

    ⚠️ 它检查的是**第一趟跑的时候**入参有哪些键（`adversarial.json.inputs_present`），
    不是「现在这一刻」——现在这一刻产物当然齐，拿它当证据等于没查。

    ⚠️ 它自己**不能在**第一趟里跑（第一趟时 `summary.json` 还没生成）。
    """
    from .pipeline import ADVERSARIAL_INPUTS
    present = set(p["adversarial.json"].get("inputs_present", []))
    if not present:
        return (False, "adversarial.json 里没有 inputs_present —— "
                       "无法证明用例是拿完整产物判的（老产物？重跑 `run`）")
    miss = sorted(k for k in ADVERSARIAL_INPUTS if k not in present)
    return (not miss, f"用例跑时 {len(present)} 份产物里需要的都在" if not miss
                     else f"跑对抗用例时缺 {miss} —— "
                          "`run` 的裁定与 `check` 的裁定会因此不同")


@_case("renderers_are_pure_functions_of_the_products",
       "渲染器只依赖产物的**内容**：同一份产物、以及它经 JSON 往返之后的副本，"
       "渲染结果逐位相同")
def _c_render_pure(p, c, gs):
    """⚠️ 关键是**第二份**：`json.loads(json.dumps(p))` 的副本。

    只比「同一份内存对象渲染两次」是**弱**断言：两次读的是同一个 dict，
    连**插入序**都一样 ⇒ 渲染器偷看键序也照样绿。
    而产物是要落盘的 —— 盘上的键序是**字典序**（`emit.dumps` 里
    `sort_keys=True`）⇒ 「从盘上重建报告」会得到另一份文本。

    本轮就是这么被抓到的：`visualize` 报「重建的报告与盘上的不同」，
    而这条用例是绿的。⇒ 判据必须把「落盘」这一步也含进去。
    """
    from . import report as R
    a = R.render(p)
    b = R.render(p)
    # 落盘再读回来的那一份（键序变成字典序、元组变列表、int 变 int）
    round_tripped = json.loads(json.dumps(p, ensure_ascii=False))
    d = R.render(round_tripped)
    if a != b:
        return (False, "同一份内存对象渲染两次就不同 —— 渲染器有内部状态")
    if a != d:
        return (False, "渲染器读了**内存里的键序**：产物经 JSON 往返后"
                       "（键序变字典序）渲染结果不同 ⇒ 从盘上重建不出同一份报告")
    return (True, "同一份、以及 JSON 往返后的副本，渲染逐位相同")


@_case("renderers_read_nothing_but_the_products",
       "渲染器**只吃产物字典**（不吃图、不吃真值、不吃运行期对象）")
def _c_render_only(p, c, gs):
    import inspect
    from . import report as R
    sig = inspect.signature(R.render)
    params = [x for x in sig.parameters if x != "products"]
    return (not params, "render(products) 只有一个入参" if not params
                        else f"多出参数 {params}")


@_case("thresholds_are_all_wired_to_code",
       "每一个门槛键都被源码**读过**（写了却没人读的门槛永远是绿的）")
def _c_wired(p, c, gs):
    src = Path(__file__).read_text(encoding="utf-8")
    miss = [k for k in c["acceptance"] if f'"{k}"' not in src]
    return (not miss, "全部门槛都被读过" if not miss else f"没人读：{miss}")


@_case("every_expectation_has_an_evaluator",
       "配置里每一条设计期望都有对应的求值函数")
def _c_exp_eval(p, c, gs):
    from . import expectations as E
    miss = [e["id"] for e in c["expectations"] if e["id"] not in E.EVALUATORS]
    return (not miss, "全部有求值函数" if not miss else f"缺：{miss}")


@_case("expectation_verdicts_are_a_partition",
       "每条期望的裁定恰好是 held / refuted / unmeasurable 三者之一，且计数对得上")
def _c_exp_part(p, c, gs):
    rows = p["expectations.json"]["rows"]
    kinds = {"held", "refuted", "unmeasurable"}
    bad = [r["id"] for r in rows if r["verdict"] not in kinds]
    e = p["expectations.json"]
    total = e["n_held"] + e["n_refuted"] + e["n_unmeasurable"]
    return (not bad and total == e["n"], f"{e['n']} 条，三分类计数 {total}")



@_case("config_round_trips_through_disk",
       "配置**原样**从磁盘读回来（写盘 → 读回，逐位相同）")
def _c_cfg_round(p, c, gs):
    """⚠️ 这条原来是 `deep_merge(DEFAULTS, c)` 与
    `deep_merge(DEFAULTS, deep_merge(DEFAULTS, c))` 相比 —— 而 `deep_merge`
    **是幂等的**，所以那条断言**恒真**。

    恒真的判据比没有判据更危险：它占着一个「这里验过了」的位置，
    而真正的往返（写盘 → 读回）一次都没发生过。

    真的往返顺带把「保险丝会不会误伤自己导出的配置」也验了：
    `dump_config` 写出来的东西必须能被 `load_config` 读回来。
    """
    import tempfile
    from .config import dump_config, load_config
    clean = {k: v for k, v in c.items() if not k.startswith("_")}
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "default.json"
        dump_config(clean, f)
        try:
            again = load_config(f)
        except Exception as e:                  # noqa: BLE001
            return (False, f"自己导出的配置读不回来：{type(e).__name__}: {e}")
    back = {k: v for k, v in again.items() if not k.startswith("_")}
    if back == clean:
        return (True, "写盘 → 读回，逐位相同")
    diff = sorted(set(clean) ^ set(back)) or [
        k for k in sorted(clean) if clean[k] != back.get(k)]
    return (False, f"往返不一致：{diff[:4]}")


@_case("protocol_digest_is_stable",
       "协议指纹由口径**唯一**决定：同一份配置算两次相同，"
       "且与产物里记的那份**互相印证**")
def _c_digest(p, c, gs):
    """⚠️ 只比「同一份 cfg 算两次」是**弱**断言 —— `protocol_digest` 就是
    一次 sha256，两次调用不可能不同。真正会出事的是**两份口径**：
    `run` 时算的那份，与落盘 `config.json.resolved` 里记的那份。
    它们不一致 ⇒ 「这个版本是按什么口径跑出来的」从此无法回答，
    而报告照样能印出一个指纹。
    """
    from .config import protocol_digest
    a, b = protocol_digest(c), protocol_digest(c)
    if a != b:
        return (False, "同一份配置算两次不同 —— 指纹不稳定")
    rec = p["config.json"]
    d = protocol_digest(rec["resolved"])
    if d != rec["protocol_digest"]:
        return (False, f"产物里记的指纹 {rec['protocol_digest']} 与按它自己的"
                       f"口径重算的 {d} 不一致 ⇒ 版本与口径对不上")
    return (True, f"稳定，且与产物里记的一致：{a}")


@_case("every_graph_declares_at_least_one_output",
       "每张图都声明了输出")
def _c_out(p, c, gs):
    bad = [g.name for g in gs if not g.outputs]
    return (not bad, "都有输出" if not bad else f"{bad}")


@_case("every_graph_is_acyclic",
       "每张图无环（拓扑排序能跑完）")
def _c_acyclic(p, c, gs):
    bad = []
    for g in gs:
        try:
            g.topo_order()
        except ValueError as e:
            bad.append(f"{g.name}: {e}")
    return (not bad, "全部无环" if not bad else "; ".join(bad))


@_case("tiles_divide_every_problem_dimension",
       "每一档 tile 都整除每一个问题维度 —— 否则余数会让「访存 = Π loops × elems」失真")
def _c_divisible(p, c, gs):
    bad = []
    for g in gs:
        for n in g.nodes:
            if n.op != "matmul":
                continue
            a = g.shape_of(n.inputs[0])
            b = g.shape_of(n.inputs[1])
            for name, tri in c["tiling_table"].items():
                if tri is None:         # `none` = 不分块，没有 tile 要整除
                    continue
                tm, tn, tk = tri
                for dim, t, tag in ((a[0], tm, "M"), (b[1], tn, "N"), (a[1], tk, "K")):
                    if dim % t != 0:
                        bad.append(f"{g.name}/{name}: {tag}={dim} 不被 {t} 整除")
    return (not bad, "全部整除" if not bad else f"{bad[:4]}")


@_case("row_reduction_kernels_span_the_full_row",
       "含 softmax/layernorm 的核，列 tile == 整行宽（否则归约不完整）")
def _c_rowspan(p, c, gs):
    """**真的读产物**，不看代码里的注释。

    ⚠️ 这条曾经写成「由 `_tile_for_kernel` 强制」并恒返回 True ——
    于是它是一句**声明**，不是一条**断言**：`_tile_for_kernel` 在
    `tile=None` 时返回 `tn=1`，行归约的列 tile 只盖住 1 列，
    而这条「不变量」照样绿。恒真的判据比没有判据更危险，
    因为它占着一个「这里验过了」的位置。
    """
    bad = []
    n_checked = 0
    for gname, rows in _ladder_rows(p).items():
        for r in rows:
            for k in r.get("kernels", []):
                if not any(op in ("softmax", "layernorm") for op in k["ops"]):
                    continue
                n_checked += 1
                if k["tile"][1] != k["out_shape"][-1]:
                    bad.append(f"{gname}@{r['name']}/{k['kid']}: "
                               f"tn={k['tile'][1]} ≠ 行宽 {k['out_shape'][-1]}")
    return (not bad and n_checked > 0,
            f"{n_checked} 个行归约核，列 tile 都盖满整行" if not bad and n_checked
            else ("产物里没有含行归约的核 —— 这条用例什么都没验到" if not n_checked
                  else f"没盖满：{bad[:3]}"))


@_case("kernel_loops_match_its_tile",
       "每个核的循环上界 == `ceil(维度 / tile)`（循环与 tile 只能有一个来源）")
def _c_loops(p, c, gs):
    """循环嵌套与 tile **必须互相推得出来**。

    ⚠️ 这条是补出来的，因为它是本轮抓到的那个真缺陷的**唯一**判据：
    `_tile_for_kernel` 给出 `tn=1`，而 `build_kernel` 里另写了一份
    「行归约 ⇒ `j` 只走一次」的特例。两个说法同时存在时，
    **静态成本按 loops 乘、解释器按 tile 跑** —— 两边差 N 倍，
    而两边各自的公式都没写错。没有这条断言，那个 bug 只能靠
    「双成本互降了」间接发现，而它会伪装成「某个公式算错了」。
    """
    bad = []
    n_checked = 0
    for gname, rows in _ladder_rows(p).items():
        for r in rows:
            for k in r.get("kernels", []):
                n_checked += 1
                tm, tn, _tk = k["tile"]
                want = (math.ceil(k["out_shape"][0] / tm),
                        math.ceil(k["out_shape"][-1] / tn))
                got = tuple(lp["extent"] for lp in k["loops"][:2])
                if got != want:
                    bad.append(f"{gname}@{r['name']}/{k['kid']}: "
                               f"loops={got} 但 tile 推出 {want}")
    return (not bad and n_checked > 0,
            f"{n_checked} 个核的循环与 tile 一致" if not bad and n_checked
            else ("没有核可查" if not n_checked else f"不一致：{bad[:3]}"))


@_case("baseline_uses_no_optimisation_pass",
       "基线（`as_written`）的核数 == 图的算子数（朴素降级确实朴素）")
def _c_base_naive(p, c, gs):
    bad = []
    for gname, rows in _ladder_rows(p).items():
        want = next(g.n_ops() for g in gs if g.name == gname)
        if rows[0]["n_kernels"] != want:
            bad.append(f"{gname}: {rows[0]['n_kernels']} vs 算子数 {want}")
    return (not bad, "每个算子一个核" if not bad else "; ".join(bad))


@_case("every_graph_has_a_matmul",
       "每张图都含至少一个矩阵乘（本项目的主战场；没有它，分块无杠杆）")
def _c_has_mm(p, c, gs):
    bad = [g.name for g in gs if not any(n.op == "matmul" for n in g.nodes)]
    return (not bad, "都有矩阵乘" if not bad else f"{bad}")


CASES: tuple[Case, ...] = tuple(
    v for k, v in sorted(globals().items()) if isinstance(v, Case)
)

#: **必须等报告能渲染之后才判**的用例。
#:
#: ⚠️ 这是一个**环**，而且环的后果不是报错，是两个数字对不上：
#: 报告要印对抗用例的结论 ⇒ `render` 要读 `adversarial.json`；
#: 而 `adversarial.json` 正是这一层在算的东西。
#: 混在一起跑的后果是这两条用例**永远「违反」**（`KeyError` 被
#: `except Exception` 兜住），而它们的声明说的是「渲染器是产物的纯函数」——
#: 判据与声明指的不是同一个对象。
#:
#: 拆法：这一层跑**两趟**。第一趟判「产物的性质」，产出的结论让
#: `summary.json` 与报告都能成型；第二趟（`pipeline` 里显式调用）判
#: 「渲染器的性质」——那时它才真的可以渲染。
REPORT_DEPENDENT_CASES: tuple[str, ...] = (
    "renderers_are_pure_functions_of_the_products",
    # ⚠️ 这条也必须在第二趟：它检查的是「第一趟跑的时候产物齐不齐」，
    #    而「齐」的定义里就有 `summary.json` —— 第一趟跑的时候它还没生成。
    "adversarial_cases_run_with_all_their_inputs_present",
)


def run_adversarial(products: dict[str, Any], cfg: dict[str, Any],
                    graphs: tuple[Graph, ...], *,
                    only: tuple[str, ...] | None = None,
                    exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    rows = []
    for case in CASES:
        if only is not None and case.name not in only:
            continue
        if case.name in exclude:
            continue
        try:
            ok, detail = case.fn(products, cfg, graphs)
        except Exception as e:                      # noqa: BLE001
            ok, detail = False, f"求值时抛错：{type(e).__name__}: {e}"
        rows.append({"name": case.name, "claim": case.claim,
                     "held": bool(ok), "detail": detail})
    out = _tally(rows)
    #: 这一趟跑的时候，产物字典里有哪些键。**记下来**，因为「用例是在产物
    #: 齐全的前提下跑的」这件事本身是一条要守的性质 —— 而它只能在**事后**
    #: 判（跑的那一刻正是要查的东西）。
    out["inputs_present"] = sorted(products)
    return out


def _tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_cases": len(rows),
        "n_held": sum(1 for r in rows if r["held"]),
        "n_violated": sum(1 for r in rows if not r["held"]),
        "cases": rows,
    }


def merge_adversarial(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """两趟的结果合成一份。**按用例名排序** ⇒ 确定性。

    `inputs_present` 取**第一趟**的：第二趟是在报告成型之后跑的，
    那时候产物当然齐 —— 拿它当证据等于没查。
    """
    rows = sorted(a["cases"] + b["cases"], key=lambda r: r["name"])
    out = _tally(rows)
    out["inputs_present"] = list(a["inputs_present"])
    return out
