"""策略空间：**交叉积展开**（不是搜索器挑的）。

`candidates are enumerated, not searched` —— 这条决定了形状判定：
候选从配置里的交叉积展开 ⇒ 不是「搜索闭环」。本项目里没有任何搜索器、
没有代理模型、没有迭代。

⚠️ 展开顺序**必须确定性**（按轴名排序 + 每轴按配置里的顺序），
否则「第 7 条策略」在不同运行里是不同东西，对比矩阵无法复现。
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable


def axes(cfg: dict[str, Any]) -> tuple[str, ...]:
    """轴名，**排序**（不是字典插入序）。"""
    return tuple(sorted(cfg["strategies"]))


def enumerate_strategies(cfg: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """全部策略，确定顺序。"""
    names = axes(cfg)
    combos: Iterable[tuple] = itertools.product(*(cfg["strategies"][n] for n in names))
    out: list[dict[str, Any]] = []
    for combo in combos:
        out.append({n: v for n, v in zip(names, combo)})
    return tuple(out)


def strategy_id(strat: dict[str, Any]) -> str:
    """稳定的策略 id。**只由轴值决定**，不含序号 —— 序号会让「同一条策略」
    在空间变化后换名字。"""
    return "s:" + "|".join(f"{k}={_short(strat[k])}" for k in sorted(strat))


def _short(v: Any) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    return str(v)


def describe(strat: dict[str, Any]) -> str:
    """给报告用的人读描述。"""
    return ", ".join(f"{k}={_short(strat[k])}" for k in sorted(strat))
