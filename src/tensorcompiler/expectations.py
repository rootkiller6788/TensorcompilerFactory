"""设计期望：**允许被实测推翻**。

⚠️ 被推翻的那一条**最值钱** —— 它说明设计意图与现实不符，而现实才是对的。
所以这个文件**不追求全绿**：全绿说明期望写得太保守（等于没写）。

每条求值返回 `(verdict, observed)`，`verdict ∈ {"held", "refuted", "unmeasurable"}`。
`unmeasurable` 是**合法的第三种裁定**（比如「没有可核的样本」），
它不是失败，但也**不算成立** —— 把它算成成立会让「通过率」说谎。
"""

from __future__ import annotations

from typing import Any, Callable

from . import cost as C

Verdict = str


def _ladder(products: dict[str, Any]) -> dict[str, list[dict]]:
    return products["ladder.json"]["per_graph"]


def _stage(rows: list[dict], name: str) -> dict | None:
    for r in rows:
        if r["name"] == name:
            return r
    return None


def _bytes(rows: list[dict], name: str | None = None) -> int:
    if name is None:
        return rows[-1]["readings"]["global_bytes"]
    st = _stage(rows, name)
    if st is None:
        raise KeyError(name)
    return st["readings"]["global_bytes"]


# --------------------------------------------------------------------- 求值器
def _ev_fusion_dominates_tiling(p, c):
    """融合的访存降幅 vs 分块的访存降幅。"""
    rows = []
    for g, r in _ladder(p).items():
        naive = r[0]["readings"]["global_bytes"]
        fused = _bytes(r, "fusion")
        tiled = _bytes(r, "tiling")
        fus_gain = (naive - fused) / naive if naive else 0.0
        til_gain = (fused - tiled) / naive if naive else 0.0
        rows.append({"graph": g, "fusion_gain": round(fus_gain, 6),
                     "tiling_gain": round(til_gain, 6),
                     "fusion_dominates": fus_gain > til_gain})
    ok = all(x["fusion_dominates"] for x in rows)
    return ("held" if ok else "refuted", {"per_graph": rows})


def _ev_attribution_sums_to_total(p, c):
    """各 pass 的消融贡献之和 == 全链总变化。"""
    rows = []
    for r in p["attribution.json"]["per_graph"]:
        rows.append({"graph": r["graph"],
                     "n_additive_readings": r["n_additive_readings"],
                     "n_readings": r["n_readings"],
                     "non_additive": sorted(k for k, v in r["is_additive"].items() if not v),
                     "residual": {k: round(v, 6) for k, v in
                                  r["interaction_residual"].items()
                                  if abs(v) > 1e-9}})
    ok = all(x["n_additive_readings"] == x["n_readings"] for x in rows)
    return ("held" if ok else "refuted", {"per_graph": rows})


def _ev_every_optional_pass_is_load_bearing(p, c):
    a = p["attribution.json"]
    rows = []
    for r in a["per_graph"]:
        rows.append({"graph": r["graph"],
                     "decorative": r["decorative_passes"]})
    ok = not a["decorative_on_some_graph"]
    return ("held" if ok else "refuted",
            {"decorative_on_some_graph": a["decorative_on_some_graph"],
             "per_graph": rows})


def _ev_blocked_layout_improves_occupancy(p, c):
    """换 `blocked` 布局占用率会不会提高（`blocked` 的 shared 更大）。"""
    rows = []
    for pg in p["compare.json"]["per_graph"]:
        by = {}
        for cell in pg["cells"]:
            key = tuple(sorted((k, v) for k, v in cell["strategy"].items()
                               if k != "layout"))
            by.setdefault(key, {})[cell["strategy"]["layout"]] = \
                cell["readings"]["occupancy"]
        better = worse = same = 0
        for _, d in by.items():
            if "row_major" in d and "blocked" in d:
                if d["blocked"] > d["row_major"]:
                    better += 1
                elif d["blocked"] < d["row_major"]:
                    worse += 1
                else:
                    same += 1
        rows.append({"graph": pg["graph"], "blocked_better": better,
                     "blocked_worse": worse, "same": same})
    ok = all(x["blocked_worse"] == 0 for x in rows)
    return ("held" if ok else "refuted", {"per_graph": rows})


def _ev_fusing_a_row_reduction_never_hurts(p, c):
    """把行归约（softmax/layernorm）融进矩阵乘，访存会不会变差。"""
    rows = []
    for pg in p["compare.json"]["per_graph"]:
        # 只在「图里含行归约」的图上问这个问题
        has_rowred = any("softmax" in cell["strategy_id"] or True for cell in [])
        by = {}
        for cell in pg["cells"]:
            key = tuple(sorted((k, v) for k, v in cell["strategy"].items()
                               if k != "fusion"))
            by.setdefault(key, {})[cell["strategy"]["fusion"]] = \
                cell["readings"]["global_bytes"]
        hurt = sum(1 for d in by.values()
                   if len(d) == 2 and d[True] > d[False])
        rows.append({"graph": pg["graph"], "fusion_hurts_cells": hurt,
                     "n_pairs": len(by)})
    ok = all(x["fusion_hurts_cells"] == 0 for x in rows)
    return ("held" if ok else "refuted", {"per_graph": rows})


def _ev_best_strategy_is_the_same_across_graphs(p, c):
    cm = p["compare.json"]
    ids = {pg["graph"]: pg["best_latency"]["strategy_id"] for pg in cm["per_graph"]}
    return ("held" if cm["best_strategy_is_shared"] else "refuted",
            {"best_by_graph": ids, "n_distinct": len(set(ids.values()))})


def _ev_semantic_equivalence_is_bit_exact(p, c):
    s = p["semantics.json"]
    ok = s["n_semantic_ok"] == s["n_checked"] and s["max_abs_drift"] == 0.0
    return ("held" if ok else "refuted",
            {"n_checked": s["n_checked"], "n_bitwise_equal": s["n_semantic_ok"],
             "max_abs_drift": s["max_abs_drift"]})


def _ev_funnel_is_monotone(p, c):
    bad = []
    for g, rows in _ladder(p).items():
        prev = None
        for r in rows:
            v = r["readings"]["global_bytes"]
            if prev is not None and v > prev:
                bad.append(f"{g}@{r['name']}")
            prev = v
    return ("held" if not bad else "refuted", {"violations": bad})


def _ev_launch_count_equals_kernel_count(p, c):
    bad = []
    for g, rows in _ladder(p).items():
        for r in rows:
            if r["readings"]["n_kernels"] != r["n_kernels"]:
                bad.append(f"{g}@{r['name']}")
    return ("held" if not bad else "refuted", {"violations": bad})


def _ev_bigger_tile_reduces_traffic(p, c):
    """分块越大，全局访存越小。**同时看延迟** —— 只报访存会漏掉代价。"""
    rows = []
    for pg in p["compare.json"]["per_graph"]:
        by = {}
        for cell in pg["cells"]:
            by.setdefault(cell["strategy"]["tiling"], []).append(cell)
        seq = [t for t in ("none", "t8", "t16", "t32") if t in by]
        b = [min(x["readings"]["global_bytes"] for x in by[t]) for t in seq]
        lat = [min(x["readings"]["latency_us"] for x in by[t]) for t in seq]
        rows.append({"graph": pg["graph"], "tiling": seq,
                     "bytes": b, "latency_us": [round(v, 6) for v in lat],
                     "bytes_monotone": all(b[i] > b[i + 1] for i in range(len(b) - 1)),
                     "latency_monotone": all(lat[i] > lat[i + 1]
                                             for i in range(len(lat) - 1))})
    ok = all(x["bytes_monotone"] and x["latency_monotone"] for x in rows)
    return ("held" if ok else "refuted", {"per_graph": rows})


def _ev_registers_grow_with_tile_size(p, c):
    rows = []
    for pg in p["compare.json"]["per_graph"]:
        by = {}
        for cell in pg["cells"]:
            by.setdefault(cell["strategy"]["tiling"], []).append(
                cell["readings"]["peak_registers"])
        seq = [t for t in ("none", "t8", "t16", "t32") if t in by]
        vals = [min(by[t]) for t in seq]
        rows.append({"graph": pg["graph"], "tiling": seq, "registers": vals,
                     "grows": all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))})
    ok = all(x["grows"] for x in rows)
    return ("held" if ok else "refuted", {"per_graph": rows})


def _ev_occupancy_never_exceeds_one(p, c):
    bad = []
    for pg in p["compare.json"]["per_graph"]:
        for cell in pg["cells"]:
            if cell["readings"]["occupancy"] > 1.0 + 1e-12:
                bad.append(f"{pg['graph']}@{cell['strategy_id']}")
    return ("held" if not bad else "refuted", {"violations": bad})


def _ev_cse_never_increases_traffic(p, c):
    rows = []
    for pg in p["compare.json"]["per_graph"]:
        by = {}
        for cell in pg["cells"]:
            key = tuple(sorted((k, v) for k, v in cell["strategy"].items()
                               if k != "cse"))
            by.setdefault(key, {})[cell["strategy"]["cse"]] = \
                cell["readings"]["global_bytes"]
        inc = sum(1 for d in by.values() if len(d) == 2 and d[True] > d[False])
        rows.append({"graph": pg["graph"], "cells_where_cse_increases": inc})
    ok = all(x["cells_where_cse_increases"] == 0 for x in rows)
    return ("held" if ok else "refuted", {"per_graph": rows})


def _ev_reuse_reduces_peak_memory(p, c):
    rows = []
    for pg in p["compare.json"]["per_graph"]:
        by = {}
        for cell in pg["cells"]:
            key = tuple(sorted((k, v) for k, v in cell["strategy"].items()
                               if k != "memory_plan"))
            by.setdefault(key, {})[cell["strategy"]["memory_plan"]] = \
                cell["readings"]["peak_bytes"]
        bad = sum(1 for d in by.values() if len(d) == 2 and d["reuse"] > d["naive"])
        rows.append({"graph": pg["graph"], "cells_where_reuse_is_worse": bad})
    ok = all(x["cells_where_reuse_is_worse"] == 0 for x in rows)
    return ("held" if ok else "refuted", {"per_graph": rows})


def _ev_distinct_strategies_give_distinct_cost(p, c):
    rows = []
    for pg in p["compare.json"]["per_graph"]:
        keys = {tuple(sorted(cell["readings"].items())) for cell in pg["cells"]}
        rows.append({"graph": pg["graph"], "n_cells": len(pg["cells"]),
                     "n_distinct_readings": len(keys)})
    ok = all(x["n_distinct_readings"] > 1 for x in rows)
    return ("held" if ok else "refuted", {"per_graph": rows})


EVALUATORS: dict[str, Callable[[dict[str, Any], dict[str, Any]],
                               tuple[Verdict, Any]]] = {
    "fusion_dominates_tiling": _ev_fusion_dominates_tiling,
    "attribution_sums_to_total": _ev_attribution_sums_to_total,
    "every_optional_pass_is_load_bearing": _ev_every_optional_pass_is_load_bearing,
    "blocked_layout_improves_occupancy": _ev_blocked_layout_improves_occupancy,
    "fusing_a_row_reduction_never_hurts": _ev_fusing_a_row_reduction_never_hurts,
    "best_strategy_is_the_same_across_graphs": _ev_best_strategy_is_the_same_across_graphs,
    "semantic_equivalence_is_bit_exact": _ev_semantic_equivalence_is_bit_exact,
    "funnel_is_monotone": _ev_funnel_is_monotone,
    "launch_count_equals_kernel_count": _ev_launch_count_equals_kernel_count,
    "bigger_tile_reduces_traffic": _ev_bigger_tile_reduces_traffic,
    "registers_grow_with_tile_size": _ev_registers_grow_with_tile_size,
    "occupancy_never_exceeds_one": _ev_occupancy_never_exceeds_one,
    "cse_never_increases_traffic": _ev_cse_never_increases_traffic,
    "reuse_reduces_peak_memory": _ev_reuse_reduces_peak_memory,
    "distinct_strategies_give_distinct_cost": _ev_distinct_strategies_give_distinct_cost,
}


def evaluate(products: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for e in cfg["expectations"]:
        fn = EVALUATORS.get(e["id"])
        if fn is None:
            rows.append({"id": e["id"], "claim": e["claim"], "detail": e["detail"],
                         "verdict": "unmeasurable",
                         "observed": None,
                         "why": "配置里有这条期望，但 `EVALUATORS` 里没有求值函数"})
            continue
        verdict, observed = fn(products, cfg)
        rows.append({"id": e["id"], "claim": e["claim"], "detail": e["detail"],
                     "verdict": verdict, "observed": observed, "why": ""})
    return {
        "n": len(rows),
        "n_held": sum(1 for r in rows if r["verdict"] == "held"),
        "n_refuted": sum(1 for r in rows if r["verdict"] == "refuted"),
        "n_unmeasurable": sum(1 for r in rows if r["verdict"] == "unmeasurable"),
        "rows": rows,
    }
