"""解释器：**真的把降级后的 Kernel IR 跑一遍**，逐次计数访存。

三条独立路径（**这是本项目「语义」这一层真值的来源**）：

1. `eval_graph` —— 直接按节点求值（不降级、不分块）。**真值**。
2. `execute(as_written 程序)` —— 朴素降级后的程序。
3. `execute(最终程序)` —— 优化后的程序。

三条必须**逐位相等**。任何一条不等，说明某个 pass 改了语义。

⚠️ **浮点加法不满足结合律**，所以「逐位相等」是一条**很强**的要求：
它要求降级**不改变归约的顺序**。本项目能做到，是因为分块是按 `k` **升序**
连续切的（`kk` 外层、`q` 内层 ⇒ `k = kk·tk + q` 升序），与未分块时
`k = 0..K-1` 的顺序**完全一致**。

⇒ 反过来说：**一旦引入「按 tile 并行归约」（每个线程各算一部分再合并），
逐位相等立刻失效**，判据就必须换成误差界。这条边界写在这里，不是偶然。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .ir import Graph, Kernel, Node, Program, Shape

#: 访存计数的键。**静态侧与动态侧共用同一套键名** —— 否则互校会
#: 拿「global.load」去比「global_load」，两边永远不等而看不出原因。
ACCESS_KEYS = ("global.load", "global.store", "shared.load", "shared.store")

T2 = list[list[float]]


@dataclass(frozen=True)
class ExecResult:
    """一次执行的产物：每个张量的值 + 逐次累加的访存计数。"""
    env: dict[str, T2]
    accesses: dict[str, int]
    n_kernel_runs: int

    def global_elems(self) -> int:
        return self.accesses.get("global.load", 0) + self.accesses.get("global.store", 0)

    def shared_elems(self) -> int:
        return self.accesses.get("shared.load", 0) + self.accesses.get("shared.store", 0)

    def to_dict(self) -> dict[str, Any]:
        return {"accesses": dict(sorted(self.accesses.items())),
                "n_kernel_runs": self.n_kernel_runs,
                "global_elems": self.global_elems(),
                "shared_elems": self.shared_elems()}


# --------------------------------------------------------------------- 标量函数
def _gelu(x: float) -> float:
    """tanh 近似（与 PyTorch 的 `gelu(approximate="tanh")` 同式）。"""
    return 0.5 * x * (1.0 + math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _softmax_row(row: list[float]) -> list[float]:
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = 0.0
    for v in ex:
        s += v
    return [v / s for v in ex]


def _layernorm_row(row: list[float], g: list[float], b: list[float],
                   eps: float = 1e-5) -> list[float]:
    n = len(row)
    mean = 0.0
    for v in row:
        mean += v
    mean /= n
    var = 0.0
    for v in row:
        d = v - mean
        var += d * d
    var /= n
    inv = 1.0 / math.sqrt(var + eps)
    return [(row[i] - mean) * inv * g[i] + b[i] for i in range(n)]


# --------------------------------------------------------------------- 真值
def eval_graph(g: Graph, feeds: dict[str, T2]) -> dict[str, T2]:
    """**真值路径**：直接按节点求值。不降级、不分块、不融合。"""
    env: dict[str, T2] = {k: [row[:] for row in v] for k, v in feeds.items()}
    for n in g.topo_order():
        env[n.output] = _apply_node(n, env, g)
    return env


def _apply_node(n: Node, env: dict[str, T2], g: Graph) -> T2:
    a = env[n.inputs[0]]
    if n.op == "matmul":
        return _matmul(a, env[n.inputs[1]])
    if n.op == "transpose":
        r, c = len(a), len(a[0])
        return [[a[i][j] for i in range(r)] for j in range(c)]
    if n.op == "add":
        if n.scalar is not None:
            return [[v + float(n.scalar) for v in row] for row in a]
        return _zip2(a, env[n.inputs[1]], lambda x, y: x + y)
    if n.op == "mul":
        if n.scalar is not None:
            return [[v * float(n.scalar) for v in row] for row in a]
        return _zip2(a, env[n.inputs[1]], lambda x, y: x * y)
    if n.op == "gelu":
        return [[_gelu(v) for v in row] for row in a]
    if n.op == "relu":
        return [[_relu(v) for v in row] for row in a]
    if n.op == "sigmoid":
        return [[_sigmoid(v) for v in row] for row in a]
    if n.op == "softmax":
        return [_softmax_row(row) for row in a]
    if n.op == "layernorm":
        gv, bv = env[n.inputs[1]][0], env[n.inputs[2]][0]
        return [_layernorm_row(row, gv, bv) for row in a]
    raise ValueError(f"未知算子 {n.op!r}")


def _matmul(a: T2, b: T2) -> T2:
    """**k 升序、逐项累加**。⚠️ 不能用 `sum()` —— Python 3.12 起
    `sum()` 对浮点用 Neumaier 补偿求和，与逐项 `+=` 的结果**不逐位相同**，
    于是「真值」与「降级后的程序」会差最后几位，而两者看起来都对。"""
    m, k = len(a), len(a[0])
    n = len(b[0])
    out: T2 = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ai = a[i]
        oi = out[i]
        for j in range(n):
            s = 0.0
            for q in range(k):
                s += ai[q] * b[q][j]
            oi[j] = s
    return out


def _zip2(a: T2, b: T2, f) -> T2:
    """逐元素二元运算，**右对齐广播**。"""
    ra, ca = len(a), len(a[0])
    rb, cb = len(b), len(b[0])
    r = max(ra, rb)
    c = max(ca, cb)
    if ra not in (1, r) or rb not in (1, r) or ca not in (1, c) or cb not in (1, c):
        raise ValueError(f"形状 ({ra},{ca}) 与 ({rb},{cb}) 不能广播")
    return [[f(a[0 if ra == 1 else i][0 if ca == 1 else j],
               b[0 if rb == 1 else i][0 if cb == 1 else j])
             for j in range(c)] for i in range(r)]


# --------------------------------------------------------------------- 执行器
def execute(program: Program, feeds: dict[str, T2],
            shapes: dict[str, Shape]) -> ExecResult:
    """把 `Program` **真的跑一遍**。

    ⚠️ 访存计数在**搬运的那一刻**累加（「数你搬了多少」，不是
    「把 loops × elems 乘出来」）—— 静态侧才是乘出来的。两条路径
    **独立实现同一个量**，所以它们必须逐位相等。
    """
    env: dict[str, T2] = {k: [row[:] for row in v] for k, v in feeds.items()}
    acc: dict[str, int] = {k: 0 for k in ACCESS_KEYS}
    for k in program.kernels:
        _run_kernel(k, env, shapes, acc)
    return ExecResult(env=env, accesses=acc, n_kernel_runs=len(program.kernels))


def _run_kernel(k: Kernel, env: dict[str, T2], shapes: dict[str, Shape],
                acc: dict[str, int]) -> None:
    tm, tn, tk = k.tile
    out_shape = k.out_shape
    rows, cols = out_shape[0], out_shape[-1]
    ni = math.ceil(rows / tm)
    nj = math.ceil(cols / tn)
    nk = k.n_reduce()

    mm = k.recipe[0] if k.recipe and k.recipe[0][0] == "matmul" else None
    result: T2 = [[0.0] * cols for _ in range(rows)]
    tiled = any(m.space == "shared" for m in k.mem)

    for i in range(ni):
        for j in range(nj):
            if mm is not None:
                aname, bname = mm[1], mm[2]
                a_all, b_all = env[aname], env[bname]
                acc_tile: T2 = [[0.0] * tn for _ in range(tm)]
                for kk in range(nk):
                    at = _slab(a_all, i * tm, tm, kk * tk, tk)
                    bt = _slab(b_all, kk * tk, tk, j * tn, tn)
                    acc["global.load"] += tm * tk + tk * tn
                    if tiled:
                        # 真的过一遍共享内存（不是只记个数）
                        sh_a = _slab(at, 0, tm, 0, tk)
                        sh_b = _slab(bt, 0, tk, 0, tn)
                        acc["shared.store"] += tm * tk + tk * tn
                        acc["shared.load"] += tm * tk + tk * tn
                    else:
                        sh_a, sh_b = at, bt
                    # ⚠️ k 升序累加：kk 外层、q 内层 ⇒ k = kk·tk + q 升序，
                    #    与未分块时 k = 0..K-1 完全同序 ⇒ 逐位相等才成立。
                    for r in range(tm):
                        ar = sh_a[r]
                        acr = acc_tile[r]
                        for c in range(tn):
                            s = acr[c]
                            for q in range(tk):
                                s += ar[q] * sh_b[q][c]
                            acr[c] = s
                post = k.recipe[1:]
            else:
                first = k.base_name
                shp = shapes[first]
                if math.prod(shp) == math.prod(out_shape) \
                        and tuple(shp) != tuple(out_shape):
                    # reshape 类（`transpose`）：输入输出维度对不齐，整块搬。
                    # 它的核**不分块**（tile = 整个输出），所以 i=j=0 只走一次。
                    if (i, j) != (0, 0):
                        raise ValueError(
                            f"kernel {k.kid}: reshape 类核不该分块，却走到了 tile ({i},{j})"
                        )
                    acc_tile = [row[:] for row in env[first]]
                    acc["global.load"] += _n_elems(acc_tile)
                else:
                    raw = _tile_of(first, env, shp, out_shape, (tm, tn), i, j)
                    # ⚠️ 计数用 **raw**（不同元素），累加用**广播后**的 tile。
                    #    两个数不能混：`_n_elems(acc_tile)` 会把广播数成复制。
                    acc["global.load"] += _n_elems(raw)
                    acc_tile = _as_tile(raw, tm, tn)
                post = k.recipe

            for step in post:
                _apply_step(step, acc_tile, env, shapes, out_shape, (tm, tn), i, j, acc)

            for r in range(tm):
                base = i * tm + r
                for c in range(tn):
                    result[base][j * tn + c] = acc_tile[r][c]
            acc["global.store"] += tm * tn

    env[k.out_name] = result


def _n_elems(t: T2) -> int:
    return sum(len(row) for row in t)


def _slab(t: T2, r0: int, nr: int, c0: int, nc: int) -> T2:
    return [t[r0 + r][c0:c0 + nc] for r in range(nr)]


def _tile_of(name: str, env: dict[str, T2], shp: Shape, out_shape: Shape,
             tile: tuple[int, int], i: int, j: int) -> T2:
    """按广播取一个输出 tile 对应的**输入子块**。

    ⚠️ 返回的是**原始子块，不做复制**。这条是双成本互校能不能对上的关键：

    广播（`bias[col]`、标量、`(1, N)` 行向量）在硬件上是**读一次、寄存器里复用**。
    返回「复制到 tm×tn 之后的形状」，`_n_elems` 就会把一次 `bias[col]`
    数成 tm 次 —— 而静态侧数的是 1 次。于是双成本互校红在一片
    「看起来只是广播」的地方，而两边各自的公式都没写错。

    ⇒ 口径写死在这里：**一次 `global.load` 的元素数 = 这个 tile 迭代真的要从
    全局取的不同元素数**。广播张量按它自己的尺寸算。
    """
    src = env[name]
    tm, tn = tile
    if len(shp) == 0:
        return [[src[0][0]]]
    if len(shp) == 1:
        # (N,) 行向量：每个 tile 只要它覆盖的那一段
        return [src[0][j * tn:j * tn + tn]]
    if len(shp) == 2:
        sr, sc = shp
        if sr not in (1, out_shape[0]) or sc not in (1, out_shape[-1]):
            # reshape 类（`transpose`）：输入输出维度对不齐，但元素总数相同。
            # ⚠️ 把它和「广播」混在一起处理，会让 transpose 的访存被算成一个
            #    恰好整除的数，**看起来对**，实际是错的。
            if math.prod(shp) == math.prod(out_shape):
                return [row[:] for row in src]
            raise ValueError(f"张量 {name} 形状 {shp} 无法切成 {out_shape} 的 tile")
        r0 = 0 if sr == 1 else i * tm
        nr = 1 if sr == 1 else tm
        c0 = 0 if sc == 1 else j * tn
        nc = 1 if sc == 1 else tn
        return _slab(src, r0, nr, c0, nc)
    raise ValueError(f"张量 {name} 形状 {shp} 不是二维 —— 本项目的 IR 只支持二维"
                     "（一维叶子输入会被广播；一维**输出**没有意义）")


def _as_tile(raw: T2, tm: int, tn: int) -> T2:
    """把**原始子块**广播成 `tm×tn` 的累加缓冲。"""
    rr, rc = len(raw), len(raw[0])
    if (rr, rc) == (tm, tn):
        return raw
    if (rr, rc) == (1, 1):
        return [[raw[0][0]] * tn for _ in range(tm)]
    if rr == 1 and rc == tn:
        return [raw[0][:] for _ in range(tm)]
    if rc == 1 and rr == tm:
        return [[raw[r][0]] * tn for r in range(tm)]
    raise ValueError(
        f"子块 {rr}×{rc} 广播不到 tile {tm}×{tn} —— "
        "本模型只支持「同形」「行向量」「列向量」「标量」四种"
    )


def _apply_binary(acc_tile: T2, raw: T2, f) -> None:
    """把 `raw` 按广播**就地**作用到 `acc_tile` 上。

    ⚠️ 广播必须在这里发生（而不是在 `_tile_of` 里预复制）——
    见 `_tile_of` 的注释：预复制会污染访存计数。
    """
    tr, tc = len(acc_tile), len(acc_tile[0])
    rr, rc = len(raw), len(raw[0])
    if (rr, rc) == (tr, tc):
        for r in range(tr):
            acc_tile[r] = [f(x, y) for x, y in zip(acc_tile[r], raw[r])]
    elif rr == 1 and rc == tc:
        seg = raw[0]
        for r in range(tr):
            acc_tile[r] = [f(x, y) for x, y in zip(acc_tile[r], seg)]
    elif (rr, rc) == (1, 1):
        v = raw[0][0]
        for r in range(tr):
            acc_tile[r] = [f(x, v) for x in acc_tile[r]]
    elif rc == 1 and rr == tr:
        for r in range(tr):
            v = raw[r][0]
            acc_tile[r] = [f(x, v) for x in acc_tile[r]]
    else:
        raise ValueError(
            f"广播失败：子块 {rr}×{rc} 对不上 tile {tr}×{tc} —— "
            "本模型只支持「同形」「行向量」「列向量」「标量」四种"
        )


def _apply_step(step: tuple, acc_tile: T2, env: dict[str, T2], shapes: dict[str, Shape],
                out_shape: Shape, tile: tuple[int, int], i: int, j: int,
                acc: dict[str, int]) -> None:
    """把配方里的一步**就地**作用在 `acc_tile` 上。"""
    op = step[0]
    if op == "gelu":
        for r, row in enumerate(acc_tile):
            acc_tile[r] = [_gelu(v) for v in row]
    elif op == "relu":
        for r, row in enumerate(acc_tile):
            acc_tile[r] = [_relu(v) for v in row]
    elif op == "sigmoid":
        for r, row in enumerate(acc_tile):
            acc_tile[r] = [_sigmoid(v) for v in row]
    elif op == "add":
        name, scalar = step[1], step[2]
        if scalar is not None:
            for r, row in enumerate(acc_tile):
                acc_tile[r] = [v + float(scalar) for v in row]
        else:
            raw = _tile_of(name, env, shapes[name], out_shape, tile, i, j)
            acc["global.load"] += _n_elems(raw)
            _apply_binary(acc_tile, raw, lambda x, y: x + y)
    elif op == "mul":
        name, scalar = step[1], step[2]
        if scalar is not None:
            for r, row in enumerate(acc_tile):
                acc_tile[r] = [v * float(scalar) for v in row]
        else:
            raw = _tile_of(name, env, shapes[name], out_shape, tile, i, j)
            acc["global.load"] += _n_elems(raw)
            _apply_binary(acc_tile, raw, lambda x, y: x * y)
    elif op == "softmax":
        for r, row in enumerate(acc_tile):
            acc_tile[r] = _softmax_row(row)
    elif op == "layernorm":
        gname, bname = step[2], step[3]
        gt = _tile_of(gname, env, shapes[gname], out_shape, tile, i, j)[0]
        bt = _tile_of(bname, env, shapes[bname], out_shape, tile, i, j)[0]
        acc["global.load"] += len(gt) + len(bt)
        for r, row in enumerate(acc_tile):
            acc_tile[r] = _layernorm_row(row, gt, bt)
    elif op == "transpose":
        # 转置核不分块 ⇒ `acc_tile` 就是整块输入，就地转置。
        # ⚠️ 必须 `[:] =` 就地改（而不是重新绑定名字）—— 重新绑定会让
        #    调用方手里那个列表对象**没被改**，症状是「转置了但结果没变」。
        rr, cc = len(acc_tile), len(acc_tile[0])
        acc_tile[:] = [[acc_tile[x][y] for x in range(rr)] for y in range(cc)]
    else:
        raise ValueError(f"配方里有未实现的步骤 {op!r}")


# --------------------------------------------------------------------- 逐位比较
def max_abs_diff(a: T2, b: T2) -> float:
    """逐位比较两个张量。**形状不同直接抛错** —— 「形状不同但元素数相同」
    不是「接近」，是**结构错了**，用误差去掩盖它会让报告说谎。"""
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError(
            f"形状不同：{len(a)}×{len(a[0])} vs {len(b)}×{len(b[0])} —— "
            "这不是「误差」，是结构错了"
        )
    d = 0.0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            d = max(d, abs(x - y))
    return d


def bitwise_equal(a: T2, b: T2) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if x != y or (x != x) != (y != y):
                return False
    return True
