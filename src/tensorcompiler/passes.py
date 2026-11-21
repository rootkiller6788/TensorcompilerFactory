"""程序级 pass：fusion / tiling / layout / memory_plan / lower。

**图级 pass 改「有哪些算子」，程序级 pass 改「每个核怎么跑」。**
两者分开的理由：报告要能回答「**这一级 lower 到底做了什么**」——
混在一起，漏斗上的每一格就说不清是谁的贡献。

`Lowering` 是**降级链的刻度**：一个不可变的中间状态。
`build(lowering, cost)` 把它落成 `Program` —— 纯函数，同样输入必得同样产物。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .ir import (ELEMENTWISE_OPS, REDUCING_OPS, Graph, Kernel, Loop, MemOp,
                 Node, Program, Shape)
from . import graph as G

#: 行归约算子：它们的**列 tile 必须盖满整行**（否则归约不完整）。
ROW_REDUCING = frozenset({"softmax", "layernorm"})


@dataclass(frozen=True)
class Lowering:
    """降级链上的一个状态。

    - `partition`：哪些节点被分到同一个核（每个元素是**节点输出名**的元组，拓扑序）；
    - `tile`：`(tm, tn, tk)`；`None` = 不分块（tile 就是整个问题）；
    - `threads`：`None` = 由 tile 推出。
    """
    graph: Graph
    partition: tuple[tuple[str, ...], ...]
    tile: tuple[int, int, int] | None = None
    layout: str = "row_major"
    memory_plan: str = "naive"
    threads: int | None = None

    def n_kernels(self) -> int:
        return len(self.partition)

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": [list(p) for p in self.partition],
            "tile": list(self.tile) if self.tile else None,
            "layout": self.layout,
            "memory_plan": self.memory_plan,
            "threads": self.threads,
        }


def initial_lowering(g: Graph) -> Lowering:
    """朴素降级：**每个算子一个核**，不分块。这是漏斗的起点。"""
    return Lowering(graph=g, partition=tuple((n.output,) for n in g.topo_order()))


# --------------------------------------------------------------------- 融合
def _fusable(pa: tuple[str, ...], pb: tuple[str, ...]) -> bool:
    """`pa` 是生产者链的算子，`pb` 是消费者链的算子。**四条规则，写清为什么**：

    1. 消费者全是**逐元素** ⇒ 可融（把后处理折进同一个核）；
    2. 消费者是 `softmax` 且生产者含 `matmul` ⇒ 可融（flash-attention 形态：
       矩阵乘的列 tile 盖满整行时，softmax 能在 tile 上直接算）；
    3. 生产者全是**逐元素**、消费者是**归约** ⇒ 可融（prologue fusion）；
    4. 其余（**归约 → 归约**）不融 —— 要在两个归约之间融合，得把中间结果
       整个物化出来，那正是融合要避免的事。
       ⇒ **注意力的两个 matmul 隔着 softmax 融不到一起**，这不是实现偷懒，
         是「归约→归约不可融」这条规则的结构性后果（真实编译器要靠 online softmax 绕开）。
    """
    if all(op in ELEMENTWISE_OPS for op in pb):
        return True
    if pb == ("softmax",) and "matmul" in pa:
        return True
    if all(op in ELEMENTWISE_OPS for op in pa) and any(op in REDUCING_OPS for op in pb):
        return True
    return False


def fuse(lowering: Lowering) -> tuple[Lowering, dict[str, Any]]:
    """融合：把**单次使用**的中间张量两侧的核并起来。

    ⚠️ 判据里有三条**必须同时成立**，少一条就会融错：
    ① 中间张量只被用一次（用两次说明有分叉，融了会重复计算）；
    ② 中间张量不是图的输出（它是输出说明外面还要看它）；
    ③ `_fusable` 的规则成立。
    """
    g = lowering.graph
    nodes = {n.output: n for n in g.topo_order()}
    order = {n.output: i for i, n in enumerate(g.topo_order())}

    uses: dict[str, int] = {}
    for n in g.nodes:
        for i in n.inputs:
            uses[i] = uses.get(i, 0) + 1
    graph_outs = set(g.outputs)

    groups: list[list[str]] = []
    owner: dict[str, int] = {}

    def ops_of(grp: list[str]) -> tuple[str, ...]:
        return tuple(nodes[o].op for o in grp)

    for n in g.topo_order():
        head_i: int | None = None
        first = n.inputs[0]
        if first in owner:
            hi = owner[first]
            if groups[hi][-1] == first:
                head_i = hi
        ok = (
            head_i is not None
            and uses.get(first, 0) == 1
            and first not in graph_outs
            and _fusable(ops_of(groups[head_i]), (n.op,))
        )
        if ok:
            groups[head_i].append(n.output)
            owner[n.output] = head_i
        else:
            groups.append([n.output])
            owner[n.output] = len(groups) - 1

    # ⚠️ 组必须按**最后一个节点的拓扑序**排 —— 线性扫描出来的顺序会把
    #    「依赖别人」的组排在前面（`dual` 里就是：先建出 [t0,t1,out]，
    #    而 `out` 依赖后面才建的 [t2,t3]）。不排序 ⇒ 执行顺序错 ⇒ 数值全错。
    groups.sort(key=lambda grp: order[grp[-1]])

    new = replace(lowering, partition=tuple(tuple(grp) for grp in groups))
    meta = {"n_kernels_before": lowering.n_kernels(),
            "n_kernels_after": new.n_kernels(),
            "fused_away": lowering.n_kernels() - new.n_kernels()}
    return new, meta


# --------------------------------------------------------------------- 分块
def apply_tiling(lowering: Lowering, name: str,
                 table: dict[str, list[int] | None]) -> tuple[Lowering, dict[str, Any]]:
    """设置 tile。`name` 是 `tiling_table` 里的键。

    ⚠️ `none` 映射到 `tile=None`（**不是 `(1,1,1)`**）—— 见 `config.tiling_table`
    的注释：这两件事的访存差 O(K) 倍。
    """
    if name not in table:
        raise ValueError(f"tiling 档位 {name!r} 不在 tiling_table 里：{sorted(table)}")
    raw = table[name]
    if raw is None:
        new = replace(lowering, tile=None)
        return new, {"tile": None}
    t = tuple(int(x) for x in raw)
    if len(t) != 3:
        raise ValueError(f"tiling_table[{name!r}] 必须是三元组 (tm, tn, tk)，实际 {t}")
    new = replace(lowering, tile=t)
    return new, {"tile": list(t)}


# --------------------------------------------------------------------- 布局
#: 布局 → `(shared 放大系数, 额外寄存器, shared 带宽倍数)`。
#:
#: ⚠️ 三个数**必须一起给**，只给前两个是一个**不完整的模型**：
#: 只写「`blocked` 要翻倍的 shared、还要多 4 个寄存器」= 这个布局只有代价、
#: 没有收益 ⇒ 「换布局一定更差」是**假设**出来的，不是**测**出来的。
#: `blocked` 存在的真实理由是把 B 的转置副本排成无 bank conflict 的形状
#: ⇒ shared 访存更快（这里记 2×），代价是 shared 占用翻倍（可能拉低占用率）。
#: ⇒ 布局轴因此是一条**真正的权衡**，方向由「shared 访存」和「占用率」
#:    哪一个在绑着决定 —— 这件事只能量，不能想。
LAYOUT_EFFECT: dict[str, tuple[float, int, float]] = {
    "row_major": (1.0, 0, 1.0),
    "blocked": (2.0, 4, 2.0),
}


def apply_layout(lowering: Lowering, name: str) -> tuple[Lowering, dict[str, Any]]:
    if name not in LAYOUT_EFFECT:
        raise ValueError(f"未知布局 {name!r}，可选 {sorted(LAYOUT_EFFECT)}")
    new = replace(lowering, layout=name)
    sc, extra, bw = LAYOUT_EFFECT[name]
    return new, {"layout": name, "shared_scale": sc,
                 "extra_registers": extra, "shared_bandwidth_scale": bw}


def apply_memory_plan(lowering: Lowering,
                      name: str) -> tuple[Lowering, dict[str, Any]]:
    if name not in ("naive", "reuse"):
        raise ValueError(f"未知内存规划 {name!r}，可选 ['naive', 'reuse']")
    new = replace(lowering, memory_plan=name)
    return new, {"memory_plan": name}


def apply_lower(lowering: Lowering, cost: dict[str, Any]) -> tuple[Lowering, dict[str, Any]]:
    """落到可执行 Kernel IR：决定**线程映射**。

    ⚠️ 这一步不改访存，只改 `threads`（进而改占用率）。「不改访存」
    所以它**不是**漏斗上「访存」那一列的台阶 —— 报告里要标出来，
    否则读者会以为「lower 这一级没做事」。

    线程数取**全局统一的一个块大小**（`cost.threads_per_block`）——
    真实编译器也是这么做的（一个 block 大小走天下，而不是逐核挑）。
    逐核挑会让「换一个核」与「换一个线程映射」两件事混在一起。
    """
    threads = int(cost["threads_per_block"])
    if threads > int(cost["max_threads_per_sm"]):
        raise ValueError(
            f"threads_per_block={threads} > max_threads_per_sm="
            f"{cost['max_threads_per_sm']} —— 一个 block 装不下"
        )
    new = replace(lowering, threads=threads)
    return new, {"threads": threads}


# --------------------------------------------------------------------- 构建
def _tile_for_kernel(lowering: Lowering, mm: Node | None,
                     out_shape: Shape, ops: tuple[str, ...]) -> tuple[int, int, int]:
    """解析这个核的 `(tm, tn, tk)`。

    四条规则，**每条都有理由**：

    1. **没有 matmul 的核不分块** —— tile 就是整个输出。
       （分块只在矩阵乘上有杠杆；给逐元素核算 tile 只是把一次访存拆成多次。）
    2. **`tile=None` 的矩阵乘 = 逐输出元素读** ⇒ `(1, 1, K)`。
       ⚠️ 这里是最容易搞反的一处：把 `tile=None` 当成「一整块装下整个问题」
       （`(M, N, K)`）会让**不分块的访存比任何分块都小** ——
       漏斗当场反向，而数字看起来完全正常（每个元素确实只读了一次）。
       「一块装下」不是「没分块」，它是**最大的一块**，而且它假设了
       A、B 能整个放进片上 —— 那正是分块要解决的事。
    3. **含行归约（softmax / layernorm）的核，列 tile 必须盖满整行** ——
       否则 softmax 只看到一行的一部分，归约不完整。这是**正确性约束**，不是性能选择。
       ⚠️ 这条对 `tile=None` **同样适用**：`(1, 1, K)` 的 `tn=1` 会让
       `nj = N`，「列 tile 盖满整行」当场不成立。
       ⇒ 不分块的行归约核是 `(1, N, K)`：**一行一行算**，这才是「没分块」在
         行归约上的含义（不是「一次算一个元素」）。
    4. 其余按配置的 tile。
    """
    rows = out_shape[0] if out_shape else 1
    cols = out_shape[-1] if out_shape else 1
    if mm is None:
        return (rows, cols, 1)
    a = lowering.graph.shape_of(mm.inputs[0])
    b = lowering.graph.shape_of(mm.inputs[1])
    M, K, N = a[0], a[1], b[1]
    if lowering.tile is None:
        return (1, N, K) if any(op in ROW_REDUCING for op in ops) else (1, 1, K)
    tm, tn, tk = lowering.tile
    if any(op in ROW_REDUCING for op in ops):
        tn = N
    return (tm, tn, tk)


def _tile_elems(shape: Shape, out_shape: Shape, tile: tuple[int, int, int]) -> int:
    """一个输出 tile 需要 `shape` 里的多少元素（**去重后**）。

    口径与 `interp._tile_of` **必须一致**，否则双成本互校会红在广播上：
    **一次 `global.load` 的元素数 = 这个 tile 迭代真的要从全局取的不同元素数**。

    - 同形 ⇒ `tm × tn`（每个输出元素读它自己那个输入元素）；
    - 广播（行向量 / 列向量 / 标量）⇒ 只数**那一段**（`tn` / `tm` / `1`）——
      广播在硬件上是读一次、寄存器里复用，不是读 `tm` 次；
    - reshape 类（`transpose`）⇒ 整块（维度对不齐但元素总数相同）。
    """
    if not shape:
        return 1
    tm, tn = tile[0], tile[1]
    if len(shape) == 1:
        return tn
    if len(shape) == 2:
        sr, sc = shape
        if sr not in (1, out_shape[0]) or sc not in (1, out_shape[-1]):
            if math.prod(shape) == math.prod(out_shape):
                return math.prod(shape)
            raise ValueError(f"形状 {shape} 与输出 {out_shape} 不能广播")
        return (1 if sr == 1 else tm) * (1 if sc == 1 else tn)
    raise ValueError(f"形状 {shape} 不是一维或二维 —— 本项目的 IR 只支持二维")


def _base_registers(ops: tuple[str, ...]) -> int:
    """寄存器基线：矩阵乘 +8（累加器 + 双索引）、行归约 +4、纯逐元素 +0。"""
    base = 8
    if "matmul" in ops:
        base += 8
    if any(op in REDUCING_OPS for op in ops):
        base += 4
    return base


def _registers_of(ops: tuple[str, ...], tm: int, tn: int, shared: int,
                  threads: int, bpe: int, extra: int) -> int:
    """寄存器压力 = 基线 + **累加器** + **片上暂存** + 布局附加。

    ⚠️ 刻意**不含 `tk`**：`tk` 是归约循环的长度，它在真实核里是一个串行循环
    （每轮复用同一批寄存器），不是「同时活着的数据」。把 `tk` 算进去会让
    「不分块」（`tk = K`，最长）拿到最大的寄存器数，于是
    「分块越大寄存器越多」这条期望被**方向相反**的假象满足。
    """
    acc = 2 * math.ceil(tm * tn / max(threads, 1))
    staged = math.ceil(shared / max(1, threads * bpe))
    return _base_registers(ops) + acc + staged + extra


def build_kernel(lowering: Lowering, group: tuple[str, ...],
                 cost: dict[str, Any]) -> Kernel:
    """把一个核（一串节点）落成 `Kernel`。"""
    g = lowering.graph
    prod = g.producers()
    nodes = [prod[o] for o in group]
    last = nodes[-1]
    out_shape = g.shape_of(last.output)
    produced = set(group)
    ops = tuple(n.op for n in nodes)

    # 外部输入（不在本核内产生）—— **按第一次出现的顺序**，确定性
    external: list[str] = []
    for n in nodes:
        for i in n.inputs:
            if i not in produced and i not in external:
                external.append(i)

    mm = next((n for n in nodes if n.op == "matmul"), None)
    tm, tn, tk = _tile_for_kernel(lowering, mm, out_shape, ops)

    # ⚠️ 循环的边界取 **`out_shape`**（这个核最终写出去的那个张量的形状），
    #    而不是「matmul 的 M/N」：两者在融合后必须相等（后处理算子不改形状），
    #    而取 out_shape 有一个额外好处 —— 它让「循环上界」与「写回索引」
    #    用的是同一个数，不可能对不上。
    rows, cols = out_shape[0], out_shape[-1]
    if mm is not None:
        M, N = g.shape_of(mm.inputs[0])[0], g.shape_of(mm.inputs[1])[1]
        if (M, N) != (rows, cols):
            raise AssertionError(
                f"kernel {'|'.join(group)}: 融合链里的 matmul 输出 {M}×{N} "
                f"与这一级的输出 {rows}×{cols} 不同 —— 后处理算子不该改形状"
            )

    # ---- 循环嵌套 ----
    # ⚠️ 循环**一律由 tile 推出**（`ceil(dim / t)`），不另写一份特例。
    #    曾经这里为「含行归约的核」硬写了一份 `[i, j=1, k]`，而 `_tile_for_kernel`
    #    在 `tile=None` 时返回 `tn=1` ⇒ `loops` 说 j 只走一次、tile 却说列宽 1。
    #    两个说法同时存在 ⇒ 静态成本按 loops 乘（少算 N 倍）、解释器按 tile 跑
    #    （多跑 N 倍），**双成本互校当场红**，而两边各自的公式都没写错。
    #    ⇒ 只留一个来源：`tn` 被钉成 N 之后 `ceil(N/N)` 自然是 1，不需要特例。
    loops = [Loop("i", math.ceil(rows / tm)), Loop("j", math.ceil(cols / tn))]
    reduce_var: str | None = None
    if mm is not None:
        loops.append(Loop("k", math.ceil(g.shape_of(mm.inputs[0])[1] / tk)))
        reduce_var = "k"
    if any(op in ROW_REDUCING for op in ops) and loops[1].extent != 1:
        raise AssertionError(
            f"kernel {'|'.join(group)}: 含行归约却把列切成 {loops[1].extent} 段 —— "
            "softmax/layernorm 只看得到一行的一部分，归约不完整"
        )

    # ---- 访存 ----
    mem: list[MemOp] = []
    tiled = lowering.tile is not None
    for name in external:
        shp = g.shape_of(name)
        if mm is not None and name in mm.inputs:
            elems = tm * tk if name == mm.inputs[0] else tk * tn
            mem.append(MemOp("global", "load", name, elems, "every"))
            # ⚠️ 只有**真的分了块**才有 shared 这一段。不分块时 A/B 直接从
            #    全局进寄存器，硬加上 shared 会凭空多出一段访存，
            #    而静态与动态会**一起错**（读同一份 mem）⇒ 互校看不出来。
            if tiled:
                mem.append(MemOp("shared", "store", name, elems, "every"))
                mem.append(MemOp("shared", "load", name, elems, "every"))
        else:
            elems = _tile_elems(shp, out_shape, (tm, tn, tk))
            mem.append(MemOp("global", "load", name, elems, "last"))
    mem.append(MemOp("global", "store", last.output, tm * tn, "last"))

    # ---- shared / registers ----
    # ⚠️ 不分块时 **shared = 0**：A/B 直接从全局进寄存器，没有中转。
    #    硬给它一个 shared_bytes（比如 (tm·tk+tk·tn)·4）会让「不存在的 shared 用量」
    #    去限制占用率，于是「不分块」被双重惩罚，而两次惩罚看起来是同一个原因。
    bpe = int(cost["bytes_per_elem"])
    shared = 0
    if tiled and mm is not None:
        shared += (tm * tk + tk * tn) * bpe
    if tiled and any(op in ROW_REDUCING for op in ops):
        shared += tn * bpe
    scale, extra, bw_scale = LAYOUT_EFFECT[lowering.layout]
    shared = int(math.ceil(shared * scale))

    threads = lowering.threads
    if threads is None:
        threads = min(int(cost["max_threads_per_sm"]), max(32, tm * tn))

    regs = _registers_of(ops, tm, tn, shared, threads, bpe, extra)

    # ---- footprint（峰值内存的原料）----
    # ⚠️ 这里用的是张量的**完整尺寸**，不是「每次访存的元素数」——
    #    「一个核活着的时候，显存里同时存在哪些张量」问的是前者。
    #    混用会让峰值内存随分块大小变，而分块**不该**影响峰值内存。
    fp = math.prod(out_shape) if out_shape else 1
    for name in external:
        shp = g.shape_of(name)
        fp += math.prod(shp) if shp else 1

    return Kernel(
        kid="k:" + "|".join(group),
        ops=ops,
        out_name=last.output,
        out_shape=out_shape,
        base_name=(mm.inputs[0] if mm is not None else nodes[0].inputs[0]),
        tile=(tm, tn, tk),
        loops=tuple(loops),
        reduce_var=reduce_var,
        mem=tuple(mem),
        shared_bytes=shared,
        registers=regs,
        threads=threads,
        footprint_elems=fp,
        recipe=tuple(_recipe_of(nodes, g)),
        shared_bandwidth_scale=bw_scale,
    )


def _recipe_of(nodes: list[Node], g: Graph) -> list[tuple]:
    """数值配方：解释器照着它真的把数算出来。

    ⚠️ 一元/归约算子**也要把输入名写进去**（`("gelu", "t0")`）——
    只写 `("gelu",)` 的话，解释器无法从配方里知道该读哪个张量，
    只能去猜，而猜错的症状是「读了一个不存在的张量」或「读错了一个」。

    ⚠️ 矩阵乘一步要把 `(M, K, N)` **写进配方**。理由不是「方便」：
    `Kernel` 手里没有图（`Program` 是可以脱离图存在的），
    所以它算工作量读数时**查不到形状**。把维度留在配方里 ⇒
    「这个核做了多少活」是一个**自足**的事实，不需要回头问图。
    """
    out: list[tuple] = []
    for n in nodes:
        if n.op == "matmul":
            a = g.shape_of(n.inputs[0])
            b = g.shape_of(n.inputs[1])
            out.append(("matmul", n.inputs[0], n.inputs[1], a[0], a[1], b[1]))
        elif n.op == "layernorm":
            out.append(("layernorm", n.inputs[0], n.inputs[1], n.inputs[2]))
        elif n.op in ("add", "mul"):
            if n.scalar is not None:
                out.append((n.op, None, n.scalar))
            else:
                out.append((n.op, n.inputs[1], None))
        else:
            out.append((n.op, n.inputs[0]))
    return out


def build(lowering: Lowering, cost: dict[str, Any]) -> Program:
    """把 `Lowering` 落成 `Program`。**纯函数**。"""
    kernels = tuple(build_kernel(lowering, grp, cost) for grp in lowering.partition)
    return Program(kernels=kernels, memory_plan=lowering.memory_plan)


# --------------------------------------------------------------------- 阶梯
#: 图级 pass（改算子集合）—— 它们之后必须**重新构建** Program。
GRAPH_PASSES = {
    "normalize": G.normalize,
    "shape_infer": G.infer_shapes,
    "const_fold": G.const_fold,
    "cse": G.cse,
}

#: 程序级 pass（改每个核怎么跑）。
PROGRAM_PASSES = ("fusion", "tiling", "layout", "memory_plan", "lower")


#: 图级 pass 的返回值 → `Stage.meta` 的键名。**逐 pass 起名**，不共用一个 `count` ——
#: 共用一个名字会让报告里「normalize 省了 1」和「cse 省了 1」看起来是同一件事。
GRAPH_PASS_META = {
    "normalize": "n_saved_ops",
    "shape_infer": "n_shapes_checked",
    "const_fold": "n_folded",
    "cse": "n_merged",
}


@dataclass(frozen=True)
class Stage:
    """降级链上的一级：名字 + 状态 + 落成的 Program + 这一级做了什么。"""
    name: str
    lowering: Lowering
    program: Program
    meta: dict[str, Any]
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "skipped": self.skipped,
            "meta": self.meta,
            "n_kernels": self.program.n_kernels(),
            "global_elems": self.program.global_elems(),
            "shared_bytes": self.program.peak_shared_bytes(),
            "registers": self.program.peak_registers(),
            "n_ops": self.lowering.graph.n_ops(),
            "partition": [list(p) for p in self.lowering.partition],
        }


def lower_ladder(graph: Graph, strat: dict[str, Any],
                 cfg: dict[str, Any]) -> list[Stage]:
    """**完整降级链**：逐级应用 pass，每一级都留下一个 `Stage`。

    `strat` 里 `False` 的开关会把对应的可选 pass 跳过（跳过也要留一级，
    否则阶梯的刻度会随策略漂移，「第 3 级是什么」就说不清了）。
    """
    cost = cfg["cost"]
    order: list[str] = list(cfg["passes"]["order"])
    optional = set(cfg["passes"]["optional"])

    g = graph
    lo = initial_lowering(g)
    stages: list[Stage] = [Stage("as_written", lo, build(lo, cost),
                                 {"note": "朴素降级：每个算子一个核，不分块"})]

    def emit(name: str, meta: dict[str, Any], *, skipped: bool = False) -> None:
        """落一级。**lowering 没变就复用上一级的 Program**。

        ⚠️ 被跳过的 pass 会留下 `skipped=True` 的一级，它的 lowering 与上一级
        完全相同；图级 pass 在图上没有可做的事时，`initial_lowering` 也会
        给出一个**与上一级相等**的 lowering（Graph 是值相等的 frozen dataclass）。
        这两种情况下 `build` 的产物必然与上一级逐字段相同 —— 再建一遍是纯开销。
        复用只依赖**相等**（`lo == prev.lowering`），不依赖「哪个 pass 叫什么名字」，
        所以不会出现「换个 pass 实现就复用错了」。
        """
        prev = stages[-1]
        prog = prev.program if (lo is prev.lowering or lo == prev.lowering) \
            else build(lo, cost)
        stages.append(Stage(name, lo, prog, meta, skipped=skipped))

    for name in order:
        enabled = name not in optional or bool(strat.get(name, True))
        if not enabled:
            emit(name, {"skipped": True}, skipped=True)
            continue
        if name in GRAPH_PASSES:
            g, count = GRAPH_PASSES[name](g)
            lo = initial_lowering(g)
            # 图变了 ⇒ 后面的程序级 pass 必须从新的朴素降级重来
            emit(name, {GRAPH_PASS_META[name]: count})
        elif name == "fusion":
            lo, meta = fuse(lo)
            emit(name, dict(meta))
        elif name == "tiling":
            lo, meta = apply_tiling(lo, strat.get("tiling", "none"),
                                    cfg["tiling_table"])
            emit(name, dict(meta))
        elif name == "layout":
            lo, meta = apply_layout(lo, strat.get("layout", "row_major"))
            emit(name, dict(meta))
        elif name == "memory_plan":
            lo, meta = apply_memory_plan(lo, strat.get("memory_plan", "naive"))
            emit(name, dict(meta))
        elif name == "lower":
            lo, meta = apply_lower(lo, cost)
            emit(name, dict(meta))
        else:
            raise ValueError(f"passes.order 里的 {name!r} 没有实现")

    return stages
