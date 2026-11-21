"""归因：**消融一个 pass**，看漏斗怎么变。

README 的验收标准第二条是「**能解释每一次 lowering 选择带来的性能差异**」。
「解释」不是写一段散文 —— 是**把那个 pass 关掉，看读数变不变**。

⚠️ 消融是「一次只关一个」，所以逐项贡献**不一定可加** ——
两个 pass 之间有交互项时，和 ≠ 总。这不是 bug，是**要报出来的事实**
（期望 `attribution_sums_to_total` 就是用来检验它的）。
"""

from __future__ import annotations

from typing import Any

from . import cost as C
from .compare import _all_off, full_strategy
from .ir import Graph
from .passes import lower_ladder

#: 浮点读数比较的相对容差。整数读数**逐位比**（用 0 容差）。
TOL = 1e-9


def _differs(a: float, b: float) -> bool:
    if isinstance(a, int) and isinstance(b, int):
        return a != b
    return abs(a - b) > TOL * max(1.0, abs(a), abs(b))


def attribute_graph(g: Graph, cfg: dict[str, Any]) -> dict[str, Any]:
    """对一张图做全 pass 消融。"""
    cost_cfg = cfg["cost"]
    full = full_strategy(cfg)
    stages_full = lower_ladder(g, full, cfg)
    prog_full = stages_full[-1].program
    c_full = C.program_cost(prog_full, cost_cfg)

    # 基线 = **朴素降级**（as_written），不是「全关」的最终结果 ——
    # 全关时 `shape_infer` / `lower` 仍然会跑，拿它当基线会把这两步的
    # （本来就为零的）贡献算进「总变化」里。
    stages_base = lower_ladder(g, _all_off(cfg), cfg)
    prog_base = stages_base[0].program
    c_base = C.program_cost(prog_base, cost_cfg)

    optional = list(cfg["passes"]["optional"])
    per_pass: dict[str, Any] = {}
    for p in optional:
        ab = dict(full)
        ab[p] = False
        stages_ab = lower_ladder(g, ab, cfg)
        c_ab = C.program_cost(stages_ab[-1].program, cost_cfg)
        delta = {k: c_ab[k] - c_full[k] for k in C.READING_KEYS}
        changed = [k for k in C.READING_KEYS if _differs(c_ab[k], c_full[k])]
        per_pass[p] = {
            "readings": {k: c_ab[k] for k in C.READING_KEYS},
            "delta_without_it": delta,
            "readings_changed": changed,
            "load_bearing": bool(changed),
            "n_kernels_without_it": c_ab["n_kernels"],
        }

    total = {k: c_base[k] - c_full[k] for k in C.READING_KEYS}
    summed = {k: sum(per_pass[p]["delta_without_it"][k] for p in optional)
              for k in C.READING_KEYS}
    additive = {k: not _differs(summed[k], total[k]) for k in C.READING_KEYS}
    residual = {k: summed[k] - total[k] for k in C.READING_KEYS}

    return {
        "graph": g.name,
        "baseline_readings": {k: c_base[k] for k in C.READING_KEYS},
        "full_readings": {k: c_full[k] for k in C.READING_KEYS},
        "total_improvement": total,
        "sum_of_ablation_deltas": summed,
        "interaction_residual": residual,
        "is_additive": additive,
        "n_additive_readings": sum(1 for v in additive.values() if v),
        "n_readings": len(C.READING_KEYS),
        "per_pass": per_pass,
        "load_bearing_passes": [p for p in optional if per_pass[p]["load_bearing"]],
        "decorative_passes": [p for p in optional if not per_pass[p]["load_bearing"]],
    }


def attribute_all(graphs: tuple[Graph, ...], cfg: dict[str, Any]) -> dict[str, Any]:
    per = [attribute_graph(g, cfg) for g in graphs]
    optional = list(cfg["passes"]["optional"])
    # 一个 pass 在**至少一张图**上有杀伤 ⇒ 它是承重的
    ever = {p: any(p in r["load_bearing_passes"] for r in per) for p in optional}
    always = {p: all(p in r["load_bearing_passes"] for r in per) for p in optional}
    return {
        "graphs": [r["graph"] for r in per],
        "n_passes": len(optional),
        "per_graph": per,
        "passes": optional,
        "load_bearing_on_some_graph": [p for p in optional if ever[p]],
        "load_bearing_on_every_graph": [p for p in optional if always[p]],
        "decorative_on_some_graph": [p for p in optional if not always[p]],
        "n_load_bearing_some": sum(1 for p in optional if ever[p]),
    }
