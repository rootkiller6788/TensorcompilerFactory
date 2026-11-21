"""确定性 RNG：**不用 `random`**。

理由：`random` 的实现在 Python 版本间会变（Mersenne Twister 的播种路径变过），
而本项目的产物要做**逐位复现**。自己写一个 64 位 LCG，行为完全可控。
"""

from __future__ import annotations

_MASK = (1 << 64) - 1
_A = 6364136223846793005
_C = 1442695040888963407


class Rng:
    """64 位 LCG。`next_float()` 落在 `[-1, 1)`。"""

    def __init__(self, seed: int) -> None:
        if seed <= 0:
            raise ValueError("seed 必须 > 0（0 不是种子，是「没设种子」）")
        self._s = seed & _MASK

    def next_u64(self) -> int:
        self._s = (self._s * _A + _C) & _MASK
        return self._s

    def next_float(self) -> float:
        """`[0,1)` 的高 53 位 → `[-1,1)`。"""
        u = self.next_u64() >> 11
        return (u / float(1 << 53)) * 2.0 - 1.0

    def matrix(self, rows: int, cols: int) -> list[list[float]]:
        return [[self.next_float() for _ in range(cols)] for _ in range(rows)]

    def vector(self, n: int) -> list[list[float]]:
        """1×n 的行向量（本项目统一用 2D 表示，标量形状是 `(n,)` → `1×n`）。"""
        return [[self.next_float() for _ in range(n)]]


def tensor_rng(seed: int, *parts: object) -> Rng:
    """由 `(seed, 部件...)` 派生一个**稳定的**子种子。

    ⚠️ 不能用 `hash()` —— 字符串的 `hash()` 每进程都不同（PYTHONHASHSEED），
    于是同样的图在两次运行里拿到不同的输入，而「产物不确定」的症状
    会伪装成「编译器不确定」。
    """
    import hashlib
    blob = "|".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")
    h = hashlib.sha256(blob).digest()
    return Rng(int.from_bytes(h[:8], "big") or 1)
