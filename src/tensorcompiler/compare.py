"""对比：策略 × 图 的矩阵 + 性能 Diff + 前沿。

**收拢点在这里**：扇出的是「降级策略」，收拢的是「一张横向可比的矩阵」。

⚠️ 「矩阵缺口 = 0」是这类形状的**第一条验收** —— 它必须是**算出来的**，
不能是「我遍历了所有策略所以当然是 0」。缺口 = 配置声明的格子数 − 实际填上的格子数。
"""

from __future__ import annotations

from typing import Any

from . import cost as C
from .ir import Graph
from .passes import lower_ladder
from .strategies import describe, enumerate_strategies, strategy_id


def _all_off(cfg: dict[str, Any]) -> dict[str, Any]:
    """把每个可选 pass 关掉。"""
    return {k: (False if isinstance(v[0], bool) else v[0])
            for k, v in cfg["strategies"].items()}


def _full_on(cfg: dict[str, Any]) -> dict[str, Any]:
    """全开：布尔取 True，枚举取**最后一项**（一般是最大的那一档）。

    ⚠️ 「最后一项」不是「最好的一项」—— 这一条由期望
    `best_strategy_is_the_same_across_graphs` 去检验，不要在这里假设。
    """
    return {k: (True if isinstance(v[0], bool) else v[-1])
            for k, v in cfg["strategies"].items()}


def full_strategy(cfg: dict[str, Any]) -> dict[str, Any]:
    return _full_on(cfg)


def baseline_strategy(cfg: dict[str, Any]) -> dict[str, Any]:
    return _all_off(cfg)


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """`a` 支配 `b`：所有目标轴 ≤，且至少一条 <。"""
    le = all(a[k] <= b[k] for k in C.OBJECTIVES)
    lt = any(a[k] < b[k] for k in C.OBJECTIVES)
    return le and lt


def pareto_front(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """前沿。**按 strategy_id 排序** ⇒ 确定性。"""
    out = []
    for i, c in enumerate(cells):
        if not any(dominates(o["readings"], c["readings"])
                   for j, o in enumerate(cells) if j != i):
            out.append(c)
    out.sort(key=lambda c: (c["readings"]["global_bytes"],
                            c["readings"]["latency_us"],
                            c["readings"]["peak_bytes"], c["strategy_id"]))
    return out


def _best(cells: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    """在 `axis` 上取最优。

    ⚠️ tie-break **必须把全部三个目标轴都用上**（最后才是 `strategy_id`）。
    只用 `(axis, strategy_id)` 会选出一个**被支配**的格：
    另一格在 `axis` 上打平、在别的轴上更小 ⇒ 它支配被选中的那一格 ⇒
    「最优延迟点必在前沿上」当场不成立，而前沿本身算得没错。
    ⇒ 判据与声明指的必须是同一个对象：`best_latency` 说的是「延迟最低」，
      但**延迟并列时谁代表它**是一个额外的选择，这个选择要跟前沿的口径一致。
    """
    order = {"latency_us": ("latency_us", "global_bytes", "peak_bytes"),
             "global_bytes": ("global_bytes", "latency_us", "peak_bytes"),
             "peak_bytes": ("peak_bytes", "global_bytes", "latency_us")}[axis]
    return min(cells, key=lambda c: tuple(c["readings"][k] for k in order)
               + (c["strategy_id"],))


def compare_graph(g: Graph, cfg: dict[str, Any],
                  program_cache: dict[tuple[str, str], tuple[Graph, Any]] | None = None
                  ) -> dict[str, Any]:
    """一张图的全部策略对比。

    `program_cache` 是可选的**显式**缓存（`(图名, 策略 id) → (最终图, 最终 Program)`）。
    `pipeline` 把同一份缓存接着交给 `check_all`：两处问的是**同一个对象**
    （每条策略的最终 Program），没有理由各降级一遍。缓存由调用方持有、
    不跨运行共享 —— 换成全局 memo 会让「这次运行用的是哪份 cfg」变得不可见。
    """
    strat_list = enumerate_strategies(cfg)
    cells: list[dict[str, Any]] = []
    for strat in strat_list:
        stages = lower_ladder(g, strat, cfg)
        last = stages[-1]
        final = last.program
        readings = C.program_cost(final, cfg["cost"])
        sid = strategy_id(strat)
        if program_cache is not None:
            program_cache[(g.name, sid)] = (last.lowering.graph, final)
        cells.append({
            "strategy_id": sid,
            "strategy": {k: strat[k] for k in sorted(strat)},
            "describe": describe(strat),
            "readings": {k: readings[k] for k in C.READING_KEYS},
            "latency_breakdown": readings["latency_breakdown"],
            "partition": [list(p) for p in last.lowering.partition],
            # ⚠️ 图级 pass 会改算子集合 ⇒ 核的划分必须与**这一级**的图对齐，
            #    不是与 `data/graphs/*.json` 里那张原图对齐。
            #    把原图的算子名拿来比对，会在 const_fold/cse 生效的格子上
            #    报「不重不漏失败」，而那 160 格其实完全正确。
            "final_nodes": sorted(n.output for n in last.lowering.graph.nodes),
        })

    # ---- 矩阵缺口（**算出来的**，不是「遍历了所以是 0」）----
    expected = 1
    for v in cfg["strategies"].values():
        expected *= len(v)
    gap = expected - len(cells)
    if gap != 0:
        raise AssertionError(
            f"矩阵缺口 = {gap}：配置声明 {expected} 格，实际填了 {len(cells)} 格 —— "
            "「遍历过了」不等于「填满了」"
        )

    base = _all_off(cfg)
    stages_base = lower_ladder(g, base, cfg)
    base_readings = C.program_cost(stages_base[-1].program, cfg["cost"])

    by_lat = _best(cells, "latency_us")
    by_byt = _best(cells, "global_bytes")
    by_mem = _best(cells, "peak_bytes")
    front = pareto_front(cells)

    # ---- 性能 Diff：每一格相对基线的降幅 ----
    for c in cells:
        r, b = c["readings"], base_readings
        c["vs_baseline"] = {
            "global_bytes_ratio": _ratio(b["global_bytes"], r["global_bytes"]),
            "latency_ratio": _ratio(b["latency_us"], r["latency_us"]),
            "peak_bytes_ratio": _ratio(b["peak_bytes"], r["peak_bytes"]),
        }

    return {
        "graph": g.name,
        "note": g.note,
        "n_ops": g.n_ops(),
        "n_strategies": len(cells),
        "matrix_expected": expected,
        "matrix_gap": gap,
        "baseline": {"strategy_id": strategy_id(base),
                     "strategy": {k: base[k] for k in sorted(base)},
                     "readings": {k: base_readings[k] for k in C.READING_KEYS}},
        "best_latency": {"strategy_id": by_lat["strategy_id"],
                         "readings": by_lat["readings"],
                         "describe": by_lat["describe"]},
        "best_bytes": {"strategy_id": by_byt["strategy_id"],
                       "readings": by_byt["readings"],
                       "describe": by_byt["describe"]},
        "best_memory": {"strategy_id": by_mem["strategy_id"],
                        "readings": by_mem["readings"],
                        "describe": by_mem["describe"]},
        "pareto": [{"strategy_id": c["strategy_id"], "describe": c["describe"],
                    "readings": c["readings"]} for c in front],
        "n_pareto": len(front),
        "cells": cells,
    }


def _ratio(base: float, now: float) -> float | None:
    if now == 0:
        return None
    return base / now


def compare_all(graphs: tuple[Graph, ...], cfg: dict[str, Any],
                program_cache: dict[tuple[str, str], tuple[Graph, Any]] | None = None
                ) -> dict[str, Any]:
    per = [compare_graph(g, cfg, program_cache) for g in graphs]
    return {
        "graphs": [p["graph"] for p in per],
        "n_graphs": len(per),
        "n_strategies": per[0]["n_strategies"] if per else 0,
        "matrix_expected_total": sum(p["matrix_expected"] for p in per),
        "matrix_gap_total": sum(p["matrix_gap"] for p in per),
        "per_graph": per,
        "best_by_graph": {p["graph"]: p["best_latency"]["strategy_id"] for p in per},
        "best_strategy_is_shared": len({p["best_latency"]["strategy_id"]
                                        for p in per}) == 1,
    }
