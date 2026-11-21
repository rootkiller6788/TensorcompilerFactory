"""IR：三种表示与访存描述。

```
Graph IR   —— 一张张量图：张量 + 算子节点（**与硬件无关**）
Program    —— 一串 Kernel（**降级链的每一级都是它**）
Kernel     —— 一个融合后的核：循环嵌套 + 每次迭代的访存 + 数值配方
```

⚠️ 三种表示**必须分开**，因为报告要回答的是「**每一级 lower 做了什么**」——
把中间表示压成一种，「降级链」就只剩首尾两点，漏斗差额无从归因。

**访存描述（`MemOp`）是静态成本与动态计数共用的唯一接口**：
- 静态侧：`Π loops × elems`（解析式，不看执行）
- 动态侧：解释器真的把循环跑一遍，逐次累加
两条**独立实现**同一个量 ⇒ 它们必须逐位相等，这就是本项目的「双成本互校」。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable

Shape = tuple[int, ...]

#: 算子集合。`ARITY` 用于形状推导与配方校验。
OPS: dict[str, int] = {
    "matmul": 2,
    "add": 2,
    "mul": 2,
    "gelu": 1,
    "relu": 1,
    "sigmoid": 1,
    "softmax": 1,
    "layernorm": 3,
    "transpose": 1,
}

#: 归约算子（它们的 kernel 里有一个归约循环）。
REDUCING_OPS: frozenset[str] = frozenset({"matmul", "softmax", "layernorm"})

#: 逐元素算子（可以任意融合进同一条链）。
ELEMENTWISE_OPS: frozenset[str] = frozenset(
    {"add", "mul", "gelu", "relu", "sigmoid"}
)


# --------------------------------------------------------------------- Graph IR
@dataclass(frozen=True)
class Tensor:
    """一个张量：名字 + 形状 + 角色。

    `role` 是**报告用的**，不参与任何计算 —— 它存在的理由是让「哪些是输入、
    哪些是常量、哪些是中间结果」在产物里可读（**输入放 data/，结论放 out/**）。
    """
    name: str
    shape: Shape
    role: str = "intermediate"      # input | const | intermediate | output

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "role": self.role}


@dataclass(frozen=True)
class Node:
    """一个算子节点。`scalar` / `axis` 是**可选属性**，不是万能袋子。"""
    op: str
    inputs: tuple[str, ...]
    output: str
    scalar: float | None = None
    axis: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op, "inputs": list(self.inputs),
                             "output": self.output}
        if self.scalar is not None:
            d["scalar"] = self.scalar
        if self.axis is not None:
            d["axis"] = self.axis
        return d


@dataclass(frozen=True)
class Graph:
    """一张张量图（Graph IR）。**与硬件无关**。"""
    name: str
    note: str
    tensors: tuple[Tensor, ...]
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]

    # ---- 索引 ----
    # ⚠️ 这三个索引**只建一次**（`cached_property` 走实例 `__dict__`，
    #    所以对 frozen dataclass 也成立）。`shape_of` / `producers` 在降级链里
    #    被调用上万次，原来的实现每次线性扫一遍张量表 —— 图小的时候看不出来，
    #    但它是「每个核 × 每个外部输入」的乘数，不是一次性成本。
    #    缓存的是**不可变图**的索引；图本身 frozen，不存在「建完就过期」。
    @cached_property
    def _tensor_map(self) -> dict[str, Tensor]:
        return {t.name: t for t in self.tensors}

    @cached_property
    def _producer_map(self) -> dict[str, Node]:
        return {n.output: n for n in self.nodes}

    @cached_property
    def _input_names(self) -> tuple[str, ...]:
        """图的**叶子**：不是任何节点输出的张量。"""
        produced = set(self._producer_map)
        return tuple(t.name for t in self.tensors if t.name not in produced)

    def tmap(self) -> dict[str, Tensor]:
        # 返回**副本**：调用方拿到的仍是一份可以随便改的字典，
        # 不会把缓存本身改坏（与缓存前的契约一致）。
        return dict(self._tensor_map)

    def shape_of(self, name: str) -> Shape:
        t = self._tensor_map.get(name)
        if t is None:
            raise KeyError(f"图 {self.name} 里没有张量 {name!r}")
        return t.shape

    def role_of(self, name: str) -> str:
        t = self._tensor_map.get(name)
        if t is None:
            raise KeyError(f"图 {self.name} 里没有张量 {name!r}")
        return t.role

    def producers(self) -> dict[str, Node]:
        return dict(self._producer_map)

    def inputs(self) -> tuple[str, ...]:
        return self._input_names

    def topo_order(self) -> tuple[Node, ...]:
        """拓扑序。**必须确定性** —— 用「就绪集合按名字排序」而不是遍历字典。"""
        produced = {n.output for n in self.nodes}
        by_out = {n.output: n for n in self.nodes}
        done: set[str] = {t.name for t in self.tensors if t.name not in produced}
        remaining = list(self.nodes)
        order: list[Node] = []
        while remaining:
            ready = [n for n in remaining
                     if all(i in done for i in n.inputs)]
            if not ready:
                missing = sorted({i for n in remaining for i in n.inputs} - done)
                raise ValueError(
                    f"图 {self.name} 有环或引用了不存在的张量：缺 {missing}"
                )
            # ⚠️ 按 output 名字排序 ⇒ 同样的图永远得到同样的拓扑序。
            #    用「列表里的下一个」会让顺序随输入文件的键序漂移。
            ready.sort(key=lambda n: n.output)
            for n in ready:
                order.append(n)
                done.add(n.output)
                remaining.remove(n)
        return tuple(order)

    def n_ops(self) -> int:
        return len(self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "note": self.note,
            "tensors": [t.to_dict() for t in self.tensors],
            "nodes": [n.to_dict() for n in self.topo_order()],
            "inputs": list(self.inputs()),
            "outputs": list(self.outputs),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Graph":
        tensors = tuple(
            Tensor(name=t["name"], shape=tuple(t["shape"]),
                   role=t.get("role", "intermediate"))
            for t in d["tensors"]
        )
        nodes = tuple(
            Node(op=n["op"], inputs=tuple(n["inputs"]), output=n["output"],
                 scalar=n.get("scalar"), axis=n.get("axis"))
            for n in d["nodes"]
        )
        return Graph(name=d["name"], note=d.get("note", ""),
                     tensors=tensors, nodes=nodes,
                     outputs=tuple(d["outputs"]))


# --------------------------------------------------------------------- 访存描述
@dataclass(frozen=True)
class MemOp:
    """**一次循环迭代**的访存。

    `when`：
    - `"every"` —— 每次迭代都发生；
    - `"last"`  —— 只在**归约循环的最后一次**迭代发生
      （累加器的写回、只在最后一个 k-tile 才用的 bias）。

    ⚠️ `when` 不能省。把写回当成 `"every"` 会让全局访存**乘上归约次数**，
    而静态成本与动态计数会**一起错**（因为它们读的是同一份 `mem`）——
    错得一致 ⇒ 互校看不出来。
    """
    space: str      # "global" | "shared"
    kind: str       # "load" | "store"
    tensor: str
    elems: int
    when: str = "every"

    def to_dict(self) -> dict[str, Any]:
        return {"space": self.space, "kind": self.kind, "tensor": self.tensor,
                "elems": self.elems, "when": self.when}


@dataclass(frozen=True)
class Loop:
    """分块级循环变量。`extent` 是**分块后的迭代次数**（不是元素数）。"""
    var: str
    extent: int

    def to_dict(self) -> dict[str, Any]:
        return {"var": self.var, "extent": self.extent}


@dataclass(frozen=True)
class Kernel:
    """一个融合后的核。

    - `tile` 是**输出 tile 的形状**（`()` = 未分块，整个输出一次算完）；
    - `loops` 是分块级循环嵌套（外层在前）；
    - `reduce_var` 指向 `loops` 里那个归约循环（`when="last"` 的判据）；
    - `recipe` 是**数值配方** —— 解释器吃它来真的把数算出来。

    ⚠️ `shared_bandwidth_scale` 是**布局的收益侧**（见 `passes.LAYOUT_EFFECT`）：
    它不改访存字节数，只改「同样多字节的 shared 访问要多久」。
    不给这个字段，布局就只剩代价，「换布局一定更差」会变成一个
    **由模型单方面决定**的结论，而不是一个**量出来**的结论。
    """
    kid: str
    ops: tuple[str, ...]
    out_name: str
    out_shape: Shape
    tile: Shape
    loops: tuple[Loop, ...]
    reduce_var: str | None
    mem: tuple[MemOp, ...]
    shared_bytes: int
    registers: int
    threads: int
    base_name: str = ""
    footprint_elems: int = 0
    recipe: tuple[tuple, ...] = ()
    shared_bandwidth_scale: float = 1.0

    # ---- 静态侧：解析式数访存（**不看执行**）----
    def n_iter(self) -> int:
        n = 1
        for lp in self.loops:
            n *= lp.extent
        return n

    def n_reduce(self) -> int:
        if self.reduce_var is None:
            return 1
        for lp in self.loops:
            if lp.var == self.reduce_var:
                return lp.extent
        raise ValueError(f"kernel {self.kid}: reduce_var={self.reduce_var!r} 不在 loops 里")

    def static_accesses(self, space: str = "global") -> int:
        """解析式：`Π loops × elems`（`last` 的要除掉归约次数）。"""
        total = 0
        for m in self.mem:
            if m.space != space:
                continue
            n = self.n_iter()
            if m.when == "last":
                n //= self.n_reduce()
            total += n * m.elems
        return total

    def static_ops_by_kind(self, space: str = "global") -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.mem:
            if m.space != space:
                continue
            n = self.n_iter()
            if m.when == "last":
                n //= self.n_reduce()
            out[m.kind] = out.get(m.kind, 0) + n * m.elems
        return out

    def n_flops(self) -> int:
        """**工作量口径**（逐算子累加，不是「这个核有多少 flops」）。

        - 矩阵乘一步：`2·M·K·N`（乘加各算一次）；
        - 其余一步：`prod(out_shape)` —— 每个输出元素记 1 次。

        ⚠️ 三条口径上的话必须写在这里，否则这个数字会被读成它不是一个东西：

        1. **不是「算力」**。`transpose` 其实不做浮点运算，但它是**纯数据搬运**，
           按输出元素数记 —— 它花的是访存时间，不是算力时间。把它记成 0 会让
           「转置是免费的」，这与成本模型的其余部分自相矛盾。
        2. **必须逐算子累加**（而不是「有 matmul 就只记 matmul」）。否则融合一个
           `gelu` 进矩阵乘核会让 FLOPs **凭空减少**，而 `flops_never_increase_
           along_the_ladder` 这条对抗用例就会变成「只要融合就红」——
           判据与声明指的不是同一个对象。
        3. **与 `tile` 无关**。同样一段数学，分块怎么切都不改变算了多少 ——
           这正是「程序级 pass 不改 FLOPs」那条不变量能成立的原因。
        """
        total = 0
        per_elem = math.prod(self.out_shape) if self.out_shape else 1
        for step in self.recipe:
            if step[0] == "matmul":
                # ("matmul", a, b, M, K, N) —— 维度**写在配方里**，
                # 因为 `Kernel` 手里没有图，它不能自己去查形状。
                _, _a, _b, m, k, n = step
                total += 2 * int(m) * int(k) * int(n)
            else:
                total += per_elem
        if not self.recipe:
            # 没有配方 = 这个核算不出数（构建期就该拦住）。报出来，不静默返回 0。
            raise ValueError(f"kernel {self.kid}: recipe 为空，无法给出工作量读数")
        return total

    def to_dict(self, *, with_recipe: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kid": self.kid,
            "ops": list(self.ops),
            "out_name": self.out_name,
            "base_name": self.base_name,
            "out_shape": list(self.out_shape),
            "tile": list(self.tile),
            "loops": [lp.to_dict() for lp in self.loops],
            "reduce_var": self.reduce_var,
            "mem": [m.to_dict() for m in self.mem],
            "shared_bytes": self.shared_bytes,
            "shared_bandwidth_scale": self.shared_bandwidth_scale,
            "registers": self.registers,
            "threads": self.threads,
            "footprint_elems": self.footprint_elems,
            "n_iter": self.n_iter(),
            "n_reduce": self.n_reduce(),
            "global_elems": self.static_accesses("global"),
            "shared_elems": self.static_accesses("shared"),
            "flops": self.n_flops(),
        }
        if with_recipe:
            d["recipe"] = [list(s) for s in self.recipe]
        return d


# --------------------------------------------------------------------- Program
@dataclass(frozen=True)
class Program:
    """一串 kernel。**降级链的每一级都是一个 `Program`** —— 这就是漏斗的刻度。

    `memory_plan` 是**程序级**的属性（不是某个核的）：它决定「不同核的 buffer
    能不能复用同一块显存」，所以它影响的是**峰值内存**，不是访存或延迟。
    """
    kernels: tuple[Kernel, ...] = field(default_factory=tuple)
    memory_plan: str = "naive"

    def __len__(self) -> int:
        return len(self.kernels)

    def __iter__(self) -> Iterable[Kernel]:
        return iter(self.kernels)

    def n_kernels(self) -> int:
        return len(self.kernels)

    def global_elems(self) -> int:
        return sum(k.static_accesses("global") for k in self.kernels)

    def global_elems_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for k in self.kernels:
            for kind, n in k.static_ops_by_kind("global").items():
                out[kind] = out.get(kind, 0) + n
        return out

    def shared_elems(self) -> int:
        return sum(k.static_accesses("shared") for k in self.kernels)

    def peak_shared_bytes(self) -> int:
        return max((k.shared_bytes for k in self.kernels), default=0)

    def peak_registers(self) -> int:
        return max((k.registers for k in self.kernels), default=0)

    def total_flops(self) -> int:
        return sum(k.n_flops() for k in self.kernels)

    def to_dict(self, *, with_recipe: bool = False) -> dict[str, Any]:
        return {
            "n_kernels": self.n_kernels(),
            "global_elems": self.global_elems(),
            "global_elems_by_kind": self.global_elems_by_kind(),
            "shared_elems": self.shared_elems(),
            "peak_shared_bytes": self.peak_shared_bytes(),
            "peak_registers": self.peak_registers(),
            "total_flops": self.total_flops(),
            "kernels": [k.to_dict(with_recipe=with_recipe) for k in self.kernels],
        }


# --------------------------------------------------------------------- 工具
def tile_grid(out_shape: Shape, tile: Shape) -> tuple[int, ...]:
    """输出按 tile 切，每个维度切几块。`tile` 为空 ⇒ 全 1。"""
    if not tile:
        return tuple(1 for _ in out_shape)
    if len(tile) != len(out_shape):
        raise ValueError(f"tile {tile} 与 out_shape {out_shape} 的维数不一致")
    return tuple(math.ceil(o / t) for o, t in zip(out_shape, tile))


def broadcast_shape(a: Shape, b: Shape) -> Shape:
    """numpy 风格右对齐广播。"""
    n = max(len(a), len(b))
    aa = (1,) * (n - len(a)) + a
    bb = (1,) * (n - len(b)) + b
    out = []
    for x, y in zip(aa, bb):
        if x == y or x == 1 or y == 1:
            out.append(max(x, y))
        else:
            raise ValueError(f"形状 {a} 与 {b} 不能广播")
    return tuple(out)
