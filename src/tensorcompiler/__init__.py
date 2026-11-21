"""TensorCompilerFactory：Tensor Graph → IR → Fusion → Kernel 的**降级链**。

形状：**直线流水线的第二形态（编译器降级链）**。
第一形态（DatasetFactory）是「过滤 / 切分」—— 对象**被筛**，漏斗差额 = 多少数据活下来；
本形态是「**重写 / 降级**」—— 对象**被改写**，漏斗差额 = 程序还要做多少活。

三层真值：
1. **语义** —— 降级前后真跑一遍，结果**逐位相等**（不是声明）；
2. **成本** —— **静态成本模型（解析式数 IR）** 与 **动态访问计数（真跑一遍）** 互校；
3. **归因** —— **消融一个 pass**，漏斗必须按预期变化。

交付的是**可执行的数字孪生**：本机没有 torch / triton / numpy，也没有可用 GPU，
所以真的把图逐级 lower 成 Kernel IR，真的按分块循环跑一遍算出数值，
真的逐次计数访存 —— 而不是把 kernel 编译出来跑。
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
