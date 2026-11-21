"""静态成本模型：**解析式**地数出访存与延迟。

⚠️ 这个文件里**每一个常数都是假设，不是实测**（都在 `config.cost` 里）。
报告必须显式标注哪些读数是算出来的 —— **把假设标成测量是这类报告最危险的失真**。

与 `interp.py` 的分工：
- `interp.py` 把循环**跑一遍**，在搬运的那一刻累加（「数你搬了多少」）；
- 本文件把 `Π loops × elems` **乘出来**（「算出来该搬多少」）。
两条**独立实现**同一个量 ⇒ 必须逐位相等（`invariants.cost_crosscheck`）。
"""

from __future__ import annotations

from typing import Any

from .ir import Kernel, Program


def occupancy_of(k: Kernel, cost: dict[str, Any]) -> float:
    """占用率。

    ⚠️ **不做 `max(1, ...)` 截断**：寄存器用超了就是「装不下一个 block」，
    占用率该掉下去。截到 1 会让「寄存器爆了」看起来和「刚刚好」一样，
    而这两件事的后果差一个数量级（后者不 spill，前者 spill）。
    """
    threads = k.threads
    if threads <= 0:
        return 0.0
    by_regs = float(cost["register_file_per_sm"]) / (k.registers * threads)
    by_shared = float(cost["shared_per_sm"]) / max(k.shared_bytes, 1)
    by_threads = float(cost["max_threads_per_sm"]) / threads
    blocks = min(by_regs, by_shared, by_threads)
    return min(1.0, blocks * threads / float(cost["max_threads_per_sm"]))


#: 占用率 → 有效带宽/算力的**模型假设**。占用率低 ⇒ 延迟藏不住 ⇒ 带宽用不满。
#: ⚠️ 这是**假设**，不是测量。换一条曲线，结论会变 —— 报告里必须写出来。
OCC_FLOOR = 0.25
OCC_SLOPE = 0.75


def _eff(cost: dict[str, Any], occ: float) -> float:
    """占用率 → 有效带宽/算力的**模型假设**。

    ⚠️ 同一台机器上，全局访存、算力、shared 访存**共用**这一条曲线
    （占用率低 ⇒ 延迟藏不住 ⇒ 三种资源都用不满）。把 shared 单独拿出来
    只会让「谁是瓶颈」这件事更难说清。
    """
    return OCC_FLOOR + OCC_SLOPE * occ


def kernel_latency_us(k: Kernel, cost: dict[str, Any]) -> float:
    """单核延迟 = launch + 全局访存 + 共享访存 + 算力（后三项按占用率打折）。"""
    occ = occupancy_of(k, cost)
    f = _eff(cost, occ)
    bw = float(cost["global_bandwidth_gb_s"]) * 1.0e3 * f      # bytes / µs
    # ⚠️ shared 带宽再乘**布局**的倍数：`blocked` 不改字节数，只改速度。
    sbw = float(cost["shared_bandwidth_gb_s"]) * 1.0e3 * f * k.shared_bandwidth_scale
    fl = float(cost["flops_per_s"]) / 1.0e6 * f                # flops / µs
    bpe = int(cost["bytes_per_elem"])
    t_mem = k.static_accesses("global") * bpe / bw
    t_sh = k.static_accesses("shared") * bpe / sbw
    t_fl = k.n_flops() / fl
    return float(cost["launch_overhead_us"]) + t_mem + t_sh + t_fl


def latency_breakdown(k: Kernel, cost: dict[str, Any]) -> dict[str, float]:
    """延迟的**四项分解** —— 报告要能回答「这个核慢在哪」。"""
    occ = occupancy_of(k, cost)
    f = _eff(cost, occ)
    bw = float(cost["global_bandwidth_gb_s"]) * 1.0e3 * f
    sbw = float(cost["shared_bandwidth_gb_s"]) * 1.0e3 * f * k.shared_bandwidth_scale
    fl = float(cost["flops_per_s"]) / 1.0e6 * f
    bpe = int(cost["bytes_per_elem"])
    return {
        "launch": float(cost["launch_overhead_us"]),
        "global_mem": k.static_accesses("global") * bpe / bw,
        "shared_mem": k.static_accesses("shared") * bpe / sbw,
        "compute": k.n_flops() / fl,
    }


def peak_bytes(program: Program, cost: dict[str, Any]) -> int:
    """峰值显存。

    - `naive`：每个核各占各的 buffer ⇒ 求和；
    - `reuse`：不同核复用同一块 ⇒ 取最大。

    ⚠️ 它**只受 memory_plan 影响**，不受分块 / 布局影响 ——
    因为 footprint 用的是张量的完整尺寸（见 `passes.build_kernel`）。
    这条不变量由 `invariants.memory_plan_only_moves_peak_bytes` 守着。
    """
    bpe = int(cost["bytes_per_elem"])
    foot = [k.footprint_elems * bpe for k in program.kernels]
    if not foot:
        return 0
    return sum(foot) if program.memory_plan == "naive" else max(foot)


def program_cost(program: Program, cost: dict[str, Any]) -> dict[str, Any]:
    """一个 `Program` 的**全部读数**。这是漏斗的一格。"""
    if not program.kernels:
        return {"n_kernels": 0, "global_elems": 0, "global_bytes": 0,
                "shared_elems": 0, "peak_shared_bytes": 0, "peak_registers": 0,
                "threads": 0, "occupancy": 0.0, "peak_bytes": 0,
                "flops": 0, "latency_us": 0.0,
                "latency_breakdown": {"launch": 0.0, "global_mem": 0.0,
                                      "shared_mem": 0.0, "compute": 0.0}}
    bpe = int(cost["bytes_per_elem"])
    occs = [occupancy_of(k, cost) for k in program.kernels]
    bd = {"launch": 0.0, "global_mem": 0.0, "shared_mem": 0.0, "compute": 0.0}
    for k in program.kernels:
        for name, v in latency_breakdown(k, cost).items():
            bd[name] += v
    return {
        "n_kernels": program.n_kernels(),
        "global_elems": program.global_elems(),
        "global_bytes": program.global_elems() * bpe,
        "global_elems_by_kind": program.global_elems_by_kind(),
        "shared_elems": program.shared_elems(),
        "peak_shared_bytes": program.peak_shared_bytes(),
        "peak_registers": program.peak_registers(),
        "threads": max((k.threads for k in program.kernels), default=0),
        # 占用率取**最小值**：整条链的吞吐由最差的那个核决定，不是平均值。
        "occupancy": min(occs) if occs else 0.0,
        "occupancy_by_kernel": {k.kid: round(o, 6) for k, o in zip(program.kernels, occs)},
        "peak_bytes": peak_bytes(program, cost),
        "flops": program.total_flops(),
        "latency_us": sum(kernel_latency_us(k, cost) for k in program.kernels),
        "latency_breakdown": {k: round(v, 9) for k, v in bd.items()},
    }


#: 一个 `Program` 的**全部读数键**。漏斗、对比矩阵、门槛都从这里取。
READING_KEYS = (
    "n_kernels", "global_elems", "global_bytes", "shared_elems",
    "peak_shared_bytes", "peak_registers", "threads", "occupancy",
    "peak_bytes", "flops", "latency_us",
)

#: 三个**越小越好**的目标轴（对比矩阵与支配关系用它）。
OBJECTIVES = ("global_bytes", "latency_us", "peak_bytes")
