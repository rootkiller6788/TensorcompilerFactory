"""图的加载与**图级**变换（normalize / shape_infer / const_fold / cse）。

图级 pass 与程序级 pass 分开的理由：图级 pass 改的是**算子集合**，
程序级 pass 改的是**每个核怎么跑**。混在一起，报告里「哪一级删掉了什么」
就说不清了。

⚠️ 图级 pass 的输出**必须重新落成 `Program`**，否则「删掉一个算子」不会反映到
漏斗上 —— 而漏斗正是本项目的主证据。这一步由 `passes.lower_ladder` 调
`initial_lowering` + `build` 完成。

⚠️ 这里**刻意不再写第二份「朴素降级」**。曾经有一份 `naive_lower`，它与
`passes.initial_lowering` + `build` 描述同一件事，但两份实现的配方格式不同
（一份把 `scalar` 放在末位、一份放在第 3 位），而**只有一份真的被跑到**。
两份实现描述同一个对象，是「同一口径」最容易被破坏的地方：
不跑的那一份不会报错，只会在某天被人当成真的用起来。
⇒ 现在只有一条路径：`initial_lowering` → `build`。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .ir import OPS as _OPS, REDUCING_OPS as _REDUCING
from .ir import Graph, Node, Tensor

# --------------------------------------------------------------------- 加载
def load_graphs(cfg: dict) -> tuple[Graph, ...]:
    """从 `data/graphs/*.json` 读全部图。**按文件名排序**（确定性）。"""
    root = Path(cfg["_root"]) if cfg.get("_root") else Path.cwd()
    d = Path(cfg["graphs_dir"])
    if not d.is_absolute():
        d = root / d
    if not d.is_dir():
        raise FileNotFoundError(f"图目录不存在：{d}")
    files = sorted(d.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"图目录里没有 *.json：{d}")
    out: list[Graph] = []
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        g = Graph.from_dict(doc)
        _validate_graph(g, f)
        out.append(g)
    return tuple(out)


def _validate_graph(g: Graph, src: Path) -> None:
    """图本身的合法性。**每条报错写清为什么** —— 文案是被断言的一部分。"""
    errs: list[str] = []
    names = [t.name for t in g.tensors]
    if len(set(names)) != len(names):
        errs.append(f"张量名有重复：{sorted(names)}")
    known = set(names)
    for n in g.nodes:
        if n.op not in _OPS:
            errs.append(f"算子 {n.op!r} 不在支持的算子表里")
            continue
        # ⚠️ `add` / `mul` 带 `scalar` 时只吃 1 个输入 —— 这是**属性改变了元数**，
        #    不是「少写了一个输入」。用固定元数表去卡它会误报。
        want = _OPS[n.op]
        if n.scalar is not None and n.op in ("add", "mul"):
            want = 1
        if len(n.inputs) != want:
            errs.append(f"{n.op} 需要 {want} 个输入，实际 {len(n.inputs)}")
        for i in n.inputs:
            if i not in known:
                errs.append(f"{n.op} 的输入 {i!r} 不在张量表里")
        if n.output not in known:
            errs.append(f"{n.op} 的输出 {n.output!r} 不在张量表里")
    for o in g.outputs:
        if o not in known:
            errs.append(f"声明的输出 {o!r} 不在张量表里")
    if not g.outputs:
        errs.append("图必须至少声明一个输出")

    # ⚠️ **节点输出必须唯一**。重复输出会让 `producers()` 静默取最后一个，
    #    而拓扑序里出现两个生产者 —— 「一个张量有几个生产者」有了两个答案。
    #    `Graph.topo_order()` 也会拒绝，但那里的报错发生在降级途中；
    #    数据文件里写错时，这一步的报错更贴近出错的源文件。
    out_counts: dict[str, int] = {}
    for n in g.nodes:
        out_counts[n.output] = out_counts.get(n.output, 0) + 1
    dup_outs = sorted(o for o, k in out_counts.items() if k > 1)
    if dup_outs:
        errs.append(f"节点输出有重复：{dup_outs} —— 一个张量只能有一个生产者")

    # ⚠️ **IR 的维数约定**：叶子（输入/常量）可以是标量 / 一维向量 / 二维矩阵
    #    （前两者用于广播），但**任何节点的输出必须是二维**。
    #    理由不是洁癖：成本模型与解释器都以「输出 tile 是 tm×tn」为前提，
    #    一维输出会被静默当成 `N×N` 或 `1×N`，而两种解释在 `constbias` 上
    #    **恰好都能算出正确的最终结果**（因为后面那次 `add` 会广播回来）。
    #    ⇒ 「今天恰好对」与「对」分不开，所以在这里把它挡住。
    produced = set(out_counts)
    leaf_roles = {"input", "const"}
    all_roles = leaf_roles | {"intermediate", "output"}
    for t in g.tensors:
        if t.role not in all_roles:
            errs.append(
                f"张量 {t.name!r} 的角色 {t.role!r} 不在 {sorted(all_roles)} 里 —— "
                "角色决定 synth_feeds 要不要给它造值，写错会推迟成真值路径的 KeyError"
            )
        if t.name in produced and len(t.shape) != 2:
            errs.append(
                f"张量 {t.name!r} 是节点输出，形状却是 {list(t.shape)} —— "
                "本项目的 IR 要求**节点输出必须是二维**"
                "（一维向量只允许作为叶子输入参与广播；"
                "要一维输出就写成 1×N）"
            )
        elif t.name not in produced:
            # 叶子只支持 0/1/2 维：解释器与成本模型都按「标量 / 行向量 / 矩阵」处理。
            # 3 维叶子会被 `synth_feeds` 静默当成 `shape[0] × shape[-1]` 的矩阵 ——
            # 数据形状与声明形状对不上，却不一定当场报错。
            if len(t.shape) > 2:
                errs.append(
                    f"叶子张量 {t.name!r} 是 {len(t.shape)} 维 {list(t.shape)} —— "
                    "本项目的 IR 只支持 0/1/2 维叶子（标量 / 向量 / 矩阵）"
                )
            if t.role not in leaf_roles:
                errs.append(
                    f"叶子张量 {t.name!r} 的角色是 {t.role!r} —— "
                    "不是任何节点的输出，就必须是 input 或 const"
                    "（否则 synth_feeds 不会给它造值）"
                )

    # 环检测（顺带检查引用顺序）
    try:
        g.topo_order()
    except ValueError as e:
        errs.append(str(e))
    if errs:
        raise ValueError(f"图 {src.name} 不合法：\n" + "\n".join(f"  - {e}" for e in errs))


# 从 ir 里取算子表（放在这里避免循环导入）
# ⚠️ 这里**不重复定义**算子表 —— 两处各写一份，改了一处忘了另一处，
#    症状是「校验通过但降级爆」，而两处看起来都对。


# --------------------------------------------------------------------- 形状
def infer_shapes(g: Graph) -> tuple[Graph, int]:
    """推导每个张量的形状，并与**声明的**形状互校。

    两条独立来源：① 数据文件里**声明**的；② 从算子语义**推导**的。
    两者不等 ⇒ 抛错。这是本项目第一处「双来源互校」。

    返回 `(新图, 互校项数)`。**不改动已声明的形状** —— 只补全缺的、检查有的。
    """
    declared = {t.name: tuple(t.shape) for t in g.tensors if t.shape}
    derived: dict[str, tuple[int, ...]] = dict(declared)
    n_checked = 0
    for n in g.topo_order():
        shp = _derive_shape(n, derived, g)
        if n.output in declared:
            if declared[n.output] != shp:
                raise ValueError(
                    f"图 {g.name}：{n.output} 声明形状 {declared[n.output]}，"
                    f"但由 {n.op}{list(n.inputs)} 推导出 {shp} —— "
                    "声明与推导必须一致（这是双来源互校的第一条）"
                )
            n_checked += 1
        derived[n.output] = shp
    tensors = tuple(
        Tensor(name=t.name, shape=derived.get(t.name, ()), role=t.role)
        for t in g.tensors
    )
    return (Graph(name=g.name, note=g.note, tensors=tensors,
                  nodes=g.nodes, outputs=g.outputs), n_checked)


def _derive_shape(n: Node, shp: dict[str, tuple[int, ...]], g: Graph) -> tuple[int, ...]:
    op = n.op
    a = shp[n.inputs[0]]
    if op == "matmul":
        b = shp[n.inputs[1]]
        if len(a) != 2 or len(b) != 2:
            raise ValueError(f"{g.name}/{n.output}: matmul 只支持 2D，实际 {a} × {b}")
        if a[1] != b[0]:
            raise ValueError(f"{g.name}/{n.output}: matmul 内维不匹配 {a} × {b}")
        return (a[0], b[1])
    if op == "transpose":
        if len(a) != 2:
            raise ValueError(f"{g.name}/{n.output}: transpose 只支持 2D，实际 {a}")
        return (a[1], a[0])
    if op == "layernorm":
        # (x, gamma, beta)：输出与 x 同形
        return a
    if op == "softmax":
        return a
    if op in ("gelu", "relu", "sigmoid"):
        return a
    if op in ("add", "mul"):
        if n.scalar is not None:
            return a
        b = shp[n.inputs[1]]
        from .ir import broadcast_shape
        return broadcast_shape(a, b)
    raise ValueError(f"未知算子 {op!r}")


# --------------------------------------------------------------------- normalize
#: 恒等标量：`mul(x, 1.0)` / `add(x, 0.0)` 是**恒等算子**。
_IDENTITY_SCALARS: dict[str, float] = {"mul": 1.0, "add": 0.0}


def normalize(g: Graph) -> tuple[Graph, int]:
    """代数简化：删掉恒等算子（`mul(x, 1.0)` / `add(x, 0.0)`）。

    ⚠️ 被删掉的张量**必须把它的使用者也改掉** —— 只删节点不改引用，
    图会指向一个不存在的张量，而错误会推迟到 lowering 才爆。
    """
    drop: dict[str, str] = {}       # 被删的输出 → 替代张量
    kept: list[Node] = []
    for n in g.topo_order():
        # 先把输入里的「被删张量」替换成它的替代者
        ins = tuple(drop.get(i, i) for i in n.inputs)
        if n.scalar is not None and n.op in _IDENTITY_SCALARS:
            if float(n.scalar) == _IDENTITY_SCALARS[n.op]:
                drop[n.output] = ins[0]
                continue
        kept.append(Node(op=n.op, inputs=ins, output=n.output,
                         scalar=n.scalar, axis=n.axis))
    # 输出也可能被删（恒等链尾）
    outs = tuple(drop.get(o, o) for o in g.outputs)
    live = {n.output for n in kept} | {o for o in outs} | set(g.inputs())
    tensors = tuple(t for t in g.tensors if t.name in live)
    n_saved = len(g.nodes) - len(kept)
    return (Graph(name=g.name, note=g.note, tensors=tensors,
                  nodes=tuple(kept), outputs=outs), n_saved)


# --------------------------------------------------------------------- const_fold
def const_fold(g: Graph) -> tuple[Graph, int]:
    """折叠**全部输入都是常量**的节点。

    折叠的判据是**角色**（`role == "const"`）而不是「看起来像常量」——
    后者需要求值，而求值需要真实数据（本项目的常量张量没有数据，
    只有形状）。⇒ 只折叠「输入全 const」的节点，输出**继承 const 角色**。
    """
    const_names = {t.name for t in g.tensors if t.role == "const"}
    foldable: dict[str, bool] = {}
    kept: list[Node] = []
    n_folded = 0
    for n in g.topo_order():
        if all(i in const_names or foldable.get(i, False) for i in n.inputs):
            foldable[n.output] = True
            n_folded += 1
            continue
        kept.append(n)
    if n_folded == 0:
        return (g, 0)
    # 被折叠的张量：如果还被别人用，保留但改成 const 角色
    still_used = {i for n in kept for i in n.inputs}
    tensors: list[Tensor] = []
    for t in g.tensors:
        if t.name in foldable:
            if t.name in still_used:
                tensors.append(Tensor(t.name, t.shape, "const"))
            continue
        tensors.append(t)
    return (Graph(name=g.name, note=g.note, tensors=tuple(tensors),
                  nodes=tuple(kept), outputs=g.outputs), n_folded)


# --------------------------------------------------------------------- cse
def cse(g: Graph) -> tuple[Graph, int]:
    """公共子表达式消除：`(op, inputs, 属性)` 相同的节点合并成一个。

    ⚠️ 键里**必须带属性**（`scalar` / `axis`）—— 只按 `(op, inputs)` 合并，
    `add(x, s=1)` 与 `add(x, s=2)` 会被当成同一个，那是**错的合并**。
    """
    seen: dict[tuple, str] = {}
    alias: dict[str, str] = {}
    kept: list[Node] = []
    n_merged = 0
    for n in g.topo_order():
        ins = tuple(alias.get(i, i) for i in n.inputs)
        key = (n.op, ins, n.scalar, n.axis)
        if key in seen:
            alias[n.output] = seen[key]
            n_merged += 1
            continue
        seen[key] = n.output
        kept.append(Node(op=n.op, inputs=ins, output=n.output,
                         scalar=n.scalar, axis=n.axis))
    outs = tuple(alias.get(o, o) for o in g.outputs)
    live = {n.output for n in kept} | set(outs) | set(g.inputs())
    live |= {i for n in kept for i in n.inputs}
    tensors = tuple(t for t in g.tensors if t.name in live)
    return (Graph(name=g.name, note=g.note, tensors=tensors,
                  nodes=tuple(kept), outputs=outs), n_merged)
