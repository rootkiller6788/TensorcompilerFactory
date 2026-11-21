# TensorCompilerFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L6 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | TensorCompilerFactory |
| 类型 | Lab |
| 真正有意思的核心 | Tensor Graph → IR → Fusion → Kernel |

---

## 二、核心命题

**这个项目不是"写一个编译器"。**

真正应该研究的是：

> Tensor Program 是怎么一步步降级成 Hardware Kernel 的？

## 三、核心主干

```
Tensor Graph
     ↓
High-Level IR
     ↓
Graph Rewrite
     ↓
Tensor IR
     ↓
Loop / Tile IR
     ↓
Memory Planning
     ↓
Kernel IR
     ↓
Code Generation
     ↓
GPU / CPU Kernel
```

## 四、整体架构

```
                 Tensor Program
                       │
                       ▼
                  Frontend
                       │
                       ▼
                  Graph IR
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
      Shape Infer   Constant Fold   Rewrite
          │            │             │
          └────────────┼─────────────┘
                       ▼
                   Tensor IR
                       │
                       ▼
                 Fusion Engine
                       │
                       ▼
                   Loop IR
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        Tile         Layout       Memory
          │            │            │
          └────────────┼────────────┘
                       ▼
                   Kernel IR
                       │
                       ▼
                 Code Generator
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
           CUDA                Triton
             │                   │
             └─────────┬─────────┘
                       ▼
                   Benchmark
```

## 五、主流水线

```
Graph
 ↓
Normalize
 ↓
Infer Shape
 ↓
Rewrite
 ↓
Fuse
 ↓
Tile
 ↓
Schedule
 ↓
Lower
 ↓
Generate Kernel
 ↓
Compile
 ↓
Benchmark
 ↓
Inspect
```

## 六、这个 Lab 最值得做的是

```
同一个算子 / 计算图
        │
        ▼
不同 lowering strategy
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
IR A   IR B   IR C
 │      │      │
 └──────┼──────┘
        ▼
Performance Diff
```

例如：

```
MatMul + Bias + GELU

Before:
MatMul
 ↓
BiasAdd
 ↓
GELU

After Fusion:
FusedMatMulBiasGELU
```

然后观察：

| 指标 | 含义 |
| --- | --- |
| Global Memory Access | 融合后减少了多少次全局访存 |
| Kernel Launch Count | 3 次 launch → 1 次 launch |
| Occupancy | 占用率变化 |
| Registers | 寄存器压力 |
| Shared Memory | 共享内存使用 |
| Latency | 端到端时延 |

**这样才是真 Compiler Lab。** 不是"我实现了一个编译器"，而是"我能解释每一次 lowering 选择带来的性能差异"。

## 七、阶段性落地计划

- **M0｜IR 骨架**：定义 Graph IR / Tensor IR 的最小表示，跑通"写一个图 → 逐级 lower 到 Kernel"（非自动）。
- **M1｜重写与融合**：实现 Shape Infer / Constant Fold / Rewrite / Fusion Engine，输出融合前后对比。
- **M2｜Tile / Layout / Memory**：实现 Loop IR 上的 tiling、layout 与 memory planning，接 Code Generator（CUDA / Triton）。
- **M3｜多策略对比与发布**：同一图的多条 lowering 路径对比，输出 Performance Diff 报告与可视化。

## 八、验收标准

能端到端完成「TensorCompilerFactory」，并在其「核心」（Tensor Graph → IR → Fusion → Kernel）上给出可复现的量化结果：

- 对 `MatMul + Bias + GELU` 等典型模式，给出融合前后的完整指标对比
- 同一计算图的多条 lowering 路径可复现地产生不同性能结果，并能解释原因

---

# 形状判定：**直线流水线的第二形态（编译器降级链）**

本工作区已经有两条「直线流水线」，它们**看起来一样、判据完全不同**：

| | 第一形态：`L0-datasetfactory` 系列 | **第二形态：本项目** |
| --- | --- | --- |
| 每一级做什么 | **筛**（过滤 / 切分 / 去重） | **改写**（重写 / 融合 / 分块 / 降级） |
| 漏斗的差额是 | 多少**数据**活下来 | 程序还要做多少**活**（访存 / 核数 / 延迟） |
| 每一级的产物 | 一份更小的数据集 | **一份新的 IR** |
| 语义约束 | 「筛掉的不许回来」 | 「改写前后**逐位相等**」 |
| 典型失效 | 漏斗不单调 | 漏斗**反向**（某个 pass 让程序做了更多访存） |

**三样家族里没有的东西**（也是本项目的全部理由）：

1. **每一级 IR 都是产物**（`ladder.json` 里每一级都带完整 `kernels`）——
   变换链**可审计**，不是「我实现了融合」而是一个数字。
2. **语义保持是可执行真值** —— 降级前后**真的各跑一遍**，逐位相等。
   不是「我保证融合不改变数值」，是「跑出来一模一样」。
3. **双成本互校 + 消融归因** —— 静态解析式与动态计数是**两条独立实现**，
   必须逐位相等；再**一次关一个 pass** 看漏斗怎么变。

**不是**：搜索闭环（策略是**枚举**的交叉积，没有任何搜索器）、
进化闭环（没有种群）、测量型（没有实测性能，见附录 B.1 的警告）。

---

# 三层真值

| 层 | 真值 | 独立实现 A | 独立实现 B | 判据 |
| --- | --- | --- | --- | --- |
| **语义** | 图的原义 | `interp.eval_graph`（在 Graph IR 上直接算） | 降级后 `interp.execute`（按 Kernel 的循环真的跑） | 逐位相等 |
| **成本** | 访存元素数 | `ir.Kernel.static_accesses`（`Π loops × elems` 解析式） | `interp.execute` 的计数器 | 逐位相等 |
| **归因** | 每个 pass 的贡献 | 全链相对朴素基线的总变化 | 一次关一个 pass 的消融之和 | 逐项可加（**允许被推翻**） |

⚠️ 「逐位相等」是一条**很强**的要求：浮点加法不满足结合律，所以它等价于
「**降级不许改变归约顺序**」。分块必须按 `k` 升序**连续**切（`kk` 外层、`q` 内层
⇒ `k = kk·tk + q` 升序）。一旦引入「按 tile 并行归约」，逐位相等立刻失效，
判据必须换成误差界 —— 本项目的 `max_semantic_drift` 门槛就是钉这一条的。

---

# 运行时核对：**门槛** vs **对抗用例**

| | 门槛（`acceptance`） | 对抗用例（`invariants`） |
| --- | --- | --- |
| 是什么 | **可调达标线** | **不可调的不变量断言** |
| 写在哪 | `configs/default.json` | 代码里（配置里没有阈值） |
| 允许失败吗 | 允许（`rc=1`，调门槛或换图） | **不允许**（红了就是缺陷） |
| 数量 | 15 条 | **48 条** |
| 能不能被「证明能红」 | 不需要 | **必须**（附录 B.9） |

⚠️ **两条都不许是恒真的**。本轮的 48 条里，有 **2 条**曾经是
「恒真」或「空转」（C.339 / C.340），是靠「逐条找能让它红的输入」找出来的。

---

# 里程碑

- **M0** IR 骨架：Graph IR / Program / Kernel 三层表示 + 朴素降级 — 已交付
- **M1** 重写与融合：normalize / shape_infer / const_fold / cse / fusion — 已交付
- **M2** Tile / Layout / Memory：tiling / layout / memory_plan / lower — 已交付
- **M3** 多策略对比与发布：64 策略 × 5 图矩阵、前沿、归因、报告、只读重建 — 已交付

---

# 附录 A｜骨架、快速开始、CLI 契约、产物

## A.1 骨架

```
L6-tensorcompilerfactory/
├── main.py                       入口（只做 sys.path 与调用 cli.main）
├── pyproject.toml                requires-python >= 3.12，dependencies = []
├── configs/default.json          口径与门槛（**代码里不硬编码阈值**）
├── data/graphs/*.json            5 张图（输入；结论放 out/）
├── src/tensorcompiler/           （19 个模块，全部只依赖标准库）
│   ├── rng.py          (  49)    确定性随机（64 位 LCG；不用内置 random / hash）
│   ├── strategies.py   (  46)    策略空间：**交叉积展开**，不是搜索
│   ├── emit.py         (  81)    规范化 JSON、内容寻址版本号、产物清单
│   ├── visualize.py    (  37)    只读重建（一行都不重跑流水线）
│   ├── attribution.py  (  99)    消融归因（一次关一个 pass + 逐项可加性）
│   ├── cost.py         ( 142)    解析成本模型（**全部是模型假设，不是实测**）
│   ├── compare.py      ( 176)    矩阵 / 前沿 / 性能 Diff
│   ├── graph.py        ( 259)    图加载、校验、三个图级 pass（normalize/const_fold/cse）
│   ├── expectations.py ( 287)    设计期望裁定（**允许被推翻**）
│   ├── config.py       ( 315)    DEFAULTS + 深合并 + 加载 + 校验 + 协议指纹
│   ├── cli.py          ( 334)    命令行（六个子命令）
│   ├── ir.py           ( 409)    Graph IR / MemOp / Loop / Kernel / Program
│   ├── report.py       ( 411)    渲染器（**只吃产物字典**）
│   ├── interp.py       ( 428)    解释器：真值 + 动态访存计数（第二层真值的独立实现）
│   ├── pipeline.py     ( 431)    编排与产物组装
│   ├── passes.py       ( 546)    降级链：每一级一个 Stage，落成 Kernel IR
│   ├── invariants.py   (1098)    **48 条对抗用例**（不可调的不变量断言）
│   └── __init__.py     (  20)
└── out/                          产物（.gitignore 里）
```

**核心路径零第三方依赖**，只用标准库。环境里**没有 torch / triton / numpy，
也没有可用 GPU** ⇒ 本项目是**数字孪生**：它把「一个 kernel 有多快」换成
「这个 kernel 要做多少访存与多少次 launch」，而**不假装量到了时间**。

**只做运行时核对**（不建 `tests/`）：关键不变量做成流水线自身的运行时核对 ——
验收门槛 + **对抗用例** + 期望表。其中对抗用例 **48 条**，
与验收门槛**分列**（口径区分见上一节）。

## A.2 快速开始

```bash
cd L6-tensorcompilerfactory
python main.py run              # 端到端跑通并落盘（~4.9 s）
python main.py check            # 只读：重跑验收 + 对抗用例（rc=1 表示有门槛未通过）
python main.py visualize        # 只读：从 out/ 重建报告并比对（~0.54 s）
python main.py verify           # 只读：确定性与产物充分性核对（~9.3 s）
python main.py show             # 只读：打印摘要
python main.py dump-config      # ⚠️ **写盘**：把当前生效的配置写回 configs/default.json
```

`--json` 时 stdout 上**只许有一段 JSON**（`main.py check --json | jq` 可用）。
**完全不写盘的四个命令**（`check` / `visualize` / `verify` / `show`）不碰任何文件；
只读命令遇缺失产物在内存里重跑并明说「本次是内存重跑，未写盘」。

## A.3 CLI 契约

| 命令 | 写盘 | 说明 |
| --- | --- | --- |
| `run` | ✅ **会写盘**（写 `out/`） | 全流水线；`--config` / `--out` 相对路径**按项目根解析** |
| `check` | ❌ **只读** | 重跑验收门槛 + 对抗用例；有任何门槛未通过则 `rc=1` |
| `visualize` | ❌ **只读**（`--target` 时才另写他处） | 从产物重建 `report.md`/`report.html` 并逐位比对 |
| `verify` | ❌ **只读**（只写临时目录，跑完删） | 两次 `run` 逐位比 + 产物充分性 |
| `show` | ❌ **只读** | 摘要 |
| `dump-config` | ✅ **会写盘**（写 `configs/default.json`） | 改完 `config.py` 的 `DEFAULTS` 后**必须**跑它（见下） |

⚠️ 「写盘」这一列用的是**两个显式标记**：`会写盘` / `只读`，**每行恰好出现一个**，
且 `--help` 里必须出现**同一个**标记。这条由 `.scratch/check_readme.py` 逐命令机器核
（同 L6-kernelfactory 的做法）。

⚠️ **`dump-config` 不是只读命令**（名字像 `dump` 而已）。它的存在理由是：
`configs/default.json` 是 `DEFAULTS` 的**落盘副本**，而 `deep_merge` 里
**列表整体替换、磁盘优先** ⇒ 磁盘上的旧列表会让代码里的新声明**静默失效**
（「改了声明、重跑，数字一字未变」）。
`load_config` 里的 `no_stale_config_keys` 只挡「磁盘上多出来的键」，
**挡不住**「磁盘上的旧列表盖掉了新列表」。
⇒ **改了配置默认值就必须重新导出**，否则「今天恰好对」与「对」无法区分。
本轮就撞过一次：把 `min_fusion_traffic_reduction` 换成 `min_fusion_kernel_reduction`
之后没重导，`run` 直接被保险丝拦下（**保险丝按设计生效**）。

⚠️ 还有一层：**`dump-config` 写回的是同一份内容时，任何基于哈希的「只读」核对
都会把它判成只读**（写回同一份 ≠ 没写）。所以「哪些命令只读」这件事
**只能读代码 / `--help` 来核，不能靠比对文件哈希来核**。

`--config` / `--out` / `--target` 的相对路径**一律按项目根解析**，与当前工作目录无关。
验证手法比「跑一遍 `run`」更强：断言 `Path(默认值).resolve()` 在
`_anchor_args` 前后**逐位不变** —— 补丁是**恒等变换**，且覆盖**所有**路径参数
在**所有**子命令上的行为。

## A.4 产物

**进版本摘要**（11 个，顺序即摘要顺序）：

| 文件 | 内容 |
| --- | --- |
| `graphs.json` | 5 张图的原始定义（图级真值的入口） |
| `ladder.json` | **两条**降级链（参考链 + 基线链），每级读数 + **完整 kernels** |
| `compare.json` | 64 策略 × 5 图 = 320 格 + 前沿 + 性能 Diff |
| `attribution.json` | 逐 pass 消融（一次关一个）+ 逐项可加性 + 交互残差 |
| `semantics.json` | 320 格的语义核对（降级后 vs 真值，**逐位**） |
| `crosscheck.json` | 320 格的**双成本互校**（静态解析式 vs 动态计数） |
| `acceptance.json` | **15 条验收门槛** + 观测值 + 阈值 + 「为什么这是错的」 |
| `adversarial.json` | **48 条对抗用例**的裁定 + `inputs_present` |
| `expectations.json` | **15 条设计期望**的裁定与证据（**允许被推翻**） |
| `config.json` | 解析后的配置 + **协议指纹**（只覆盖口径，不含路径与时间） |
| `summary.json` | 头条数字 |

**落盘但不进摘要**：`version.json`（唯一允许印时间戳的地方）、
`benchmark.json`（**实测耗时**）、`report.md` / `report.html`（视图，正文里印了 `created_at`）。

⇒ 于是「同一份输入跑两次」要**精确表述**为：
**15 个产物里 11 个逐位相同；只有 4 个不同 ——
`version.json`（时间戳）、`benchmark.json`（实测耗时）、
`report.md` / `report.html`（正文里印了 `created_at`）**。
这条由 `python main.py verify` 逐位核对。

---

# 附录 B｜实测结果

> 本节每一个数字都从**最终 `out/`** 重新导出，不是凭记忆或早期运行写的。
> 配置：`seed=20261002`、5 张图、9 个 pass（可选 7）、64 条策略、
> 协议指纹 `365e17d9fd5d410e`、版本 `v1-7f10d07f8e8b`。

## B.0 一张表

| 项 | 值 |
| --- | --- |
| 图 × 策略 = 矩阵 | 5 × 64 = **320 格**，缺口 **0** |
| 降级链 | **10 级**（`as_written` + 9 个 pass），另加一条基线链 |
| 语义核对 | **320/320 逐位相等**，`max_abs_drift = 0.0` |
| 双成本互校 | **320/320 一致**（global 320、shared 320） |
| 验收门槛 | **15/15 通过** |
| 对抗用例 | **48/48 成立**（且 48/48 **都能被证明能红**） |
| 设计期望 | **9 成立 / 6 被推翻 / 0 算不出来** |
| 前沿点合计 | **20** 个 |

## B.1 规模与耗时

| 项 | 值 |
| --- | --- |
| `run` 端到端 | **4.9 s**（进程墙钟） |
| `verify`（两次 run + 逐位比 + 充分性） | 9.3 s |
| `visualize`（从产物重建，**不跑流水线**） | **0.54 s** |
| `check`（重跑验收 + 48 条对抗） | 0.61 s |
| 变异测试（48 条 × 通用变异池） | 34.2 s |

`run` 的内部**分项**耗时写在 `benchmark.json` 里，但**故意不往这里抄**：
它每次运行都不同、也不进版本摘要。能说的稳定事实是
「`ladder`（**两条链**，每级真跑）与 `semantics_and_cost`（320 格 × 真跑两遍）
合计占九成以上」。

⚠️ 加第二条降级链的代价是可见的：`ladder` 这一段的耗时**翻了一倍**。
买到的是一条**非空转**的断言（C.340）。

⚠️ **这些耗时只是孪生自身一致性的读数，不是性能读数。** 本机没有 GPU，
「这个 kernel 有多快」这个数量不出来。产物里的 `latency_us` 是
**成本模型的预测**（`Σ(launch + 访存 + 算力)`），不是测量 ——
判据 `no_elapsed_time_key_in_the_versioned_products` 用一个**短白名单**
（`latency_us` / `launch_overhead_us`）把这件事钉死：白名单一旦变长，
就说明「模型预测」与「实测耗时」在产物里混住了。

## B.2 漏斗：`mlp3` 的逐级降级

```
级            核数 算子  全局访存元素   shared  寄存器  线程  占用率   延迟 µs
as_written     4    4      138272        0      22    1024   1.00    13.056
normalize      3    3      136224        0      22    1024   1.00    10.041
shape_infer    3    3      136224        0      22    1024   1.00    10.041
const_fold     3    3      136224        0      22    1024   1.00    10.041
cse            3    3      136224        0      22    1024   1.00    10.041
fusion         1    3      133120        0      22      32   1.00     4.021
tiling         1    3        5152     8192      24    1024   1.00     3.176
layout         1    3        5152    16384      30    1024   1.00     3.172
memory_plan    1    3        5152    16384      30    1024   1.00     3.172
lower          1    3        5152    16384      48     256   0.75     3.211
```

**读法**（每一列都在说不同的事）：

- **核数**：`4 → 3` 是 `normalize`（`mul(x, 1.0)` 被删掉 ⇒ 少一个核），
  `3 → 1` 是 `fusion`（三个算子融进一个核）。**融合的价值主要在核数上**。
- **访存**：`553088 → 20608`（**26.8×**）。其中 `fusion` 只贡献 `553088 → 532480`
  （1.04×），`tiling` 贡献 `532480 → 20608`（**25.8×**）。
  ⇒ 这就是期望 `fusion_dominates_tiling` 被**推翻**的地方（B.7）。
- **线程 / 占用率**：`fusion` 把线程数从 1024 掉到 **32**（一个核只干原来四分之一
  的活），`tiling` 把它拉回 1024；`lower` 再按 `threads_per_block = 256`
  落到 256 ⇒ 占用率 `1.00 → 0.75`。
- **延迟**：`13.056 → 3.211`（4.07×）。注意最后一级 `lower` 让延迟**回升**
  （3.172 → **3.211**）：占用率从 1.00 掉到 0.75，`_eff` 系数变小 ⇒ 有效带宽变差。
  **「降级链上每一级都变好」是假的。**
- **shared / 寄存器**：`layout` 把它们从 `(8192, 24)` 抬到 `(16384, 30)`，
  而**延迟几乎没动**（3.176 → 3.172，−0.004 µs）—— 这就是 B.6 里
  `layout` 的 Δ 为负、期望 `blocked_layout_improves_occupancy` 被推翻的来源。

## B.3 语义：**320/320 逐位相等**

| 项 | 值 |
| --- | --- |
| 检查格数 | 320 |
| 逐位相等 | **320** |
| `max_abs_drift` | **0.0** |

⚠️ 这里的「0.0」**不是**「差得很小」：判据是 `bitwise_equal`
（`struct.pack("<d", a) == struct.pack("<d", b)`）。浮点加法不满足结合律，
所以这条等价于「**降级没有改变归约顺序**」。

`interp._matmul` 是**逐项 `k` 升序累加**（不用 `sum()`）——
用 `sum()` 会走不同的累加路径，`dual` 图上的逐位相等立刻红。

## B.4 双成本互校：**320/320**

两条**独立实现**算同一个量：

| | 实现 | 口径 |
| --- | --- | --- |
| 静态 | `Kernel.static_accesses(space)` | `Σ over MemOp: Π loops × elems`（解析式，不看执行） |
| 动态 | `interp.execute` 的计数器 | 把循环**真的跑一遍**，逐次累加 |

| 项 | 值 |
| --- | --- |
| 检查格数 | 320 |
| 全局访存一致 | **320** |
| shared 访存一致 | **320** |

⚠️ 这条核对是本项目**最值钱的一条**：本轮它一次抓出了 **3 个真缺陷**
（C.324 / C.325 / C.326），而每一个的**症状都是「某个公式算错了」**——
没有这条对照物，它们会被归成「公式抄错了」，而实际上是
「**循环上界有两个来源**」这种结构性问题。

## B.5 对比矩阵与前沿

| 图 | 算子数 | 格数 | 前沿点数 | 基线访存 | 最优访存 | 基线延迟 µs | 最优延迟 µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `attn` | 4 | 64 | 2 | 1085440 | 61440 | 14.074 | 9.384 |
| `constbias` | 3 | 64 | 6 | 536704 | 20608 | 7.027 | 3.175 |
| `dual` | 5 | 64 | 2 | 1085568 | 41088 | 17.074 | 6.357 |
| `mlp3` | 4 | 64 | 6 | 544896 | 20608 | 10.041 | 3.176 |
| `normmlp` | 4 | 64 | 4 | 561792 | 37504 | 13.071 | 6.206 |

⚠️ 这里的「基线」是 `_all_off(cfg)`（布尔轴全 `False`、枚举轴取各自第一项），
**不是** `as_written` —— 两者不同：`as_written` 是「朴素降级但 pass 都跑」，
基线是「pass 都不跑」。口径不一致就会得到两个都叫「基线」的不同数字。

**最优策略跨图不唯一**（`best_strategy_is_shared = False`）：

```
attn       s:cse=off|fusion=on|layout=row_major|memory_plan=reuse|tiling=t32
constbias  s:cse=off|fusion=on|layout=row_major|memory_plan=naive|tiling=t32
dual       s:cse=off|fusion=on|layout=row_major|memory_plan=reuse|tiling=t32
mlp3       s:cse=off|fusion=on|layout=row_major|memory_plan=naive|tiling=t32
normmlp    s:cse=off|fusion=on|layout=row_major|memory_plan=reuse|tiling=t32
```

5 张图上有 **2 条**不同的最优策略，**只差 `memory_plan` 一根轴**。
而 5 张图上**一致**的部分是：`cse=off`、`fusion=on`、`layout=row_major`、`tiling=t32`。

⚠️ `cse=off` 出现在**最优**里，看起来反直觉（CSE 不是优化吗？）。要分两半说：

- **4 张图（`attn` / `constbias` / `mlp3` / `normmlp`）上 `cse` 是恒等变换** ——
  它们没有重复子表达式，`cse=on` 与 `cse=off` 给出**完全相同的成本向量**。
  最优里写 `cse=off` 只是**并列时的字典序**（`_best` 的 tie-break），不是结论。
- **`dual` 上它是一条真结论**：`cse` 把算了两遍的 `matmul(A, B)` 合并 ⇒
  工作量 `265216 → 134144`（降 **49%**）、访存 `45184 → 41088`（降 9%），
  但**核数 `2 → 3`** ⇒ `launch` 从 6.0 µs 涨到 9.0 µs ⇒
  延迟 **6.357 → 9.211 µs**（涨 **45%**）。

  为什么核数会涨？因为 `cse` 让 `t0` 被**用了两次**，而融合的前提是
  「中间张量**只被用一次**」（`fuse` 的三条判据之一）⇒
  融合被拆开了。**CSE 减少计算，同时制造扇出、破坏融合**（C.346）。

⚠️ 「**优化 pass 不总是让目标变好**」这件事本身是产出。而它只有在
「目标轴不止一条」时才看得见 —— 只看 `flops` 会得出**相反**的结论。

## B.6 归因：一次关一个 pass

`delta_without_it[k]` = **关掉这个 pass 之后读数会变差多少**（正数 = 这个 pass 有用）。

| pass | attn 访存Δ | attn 延迟Δ | constbias 访存Δ | dual 访存Δ | mlp3 访存Δ | normmlp 访存Δ |
| --- | --- | --- | --- | --- | --- | --- |
| `normalize` | 0 | 0 | 0 | 0 | 0 | 0 |
| `const_fold` | 0 | 0 | 384 | 0 | 0 | 0 |
| `cse` | 0 | 0 | 0 | 4096 | 0 | 0 |
| `fusion` | 8192 | 3.011 | 8192 | 8192 | 16384 | 16384 |
| `tiling` | 761856 | 1.179 | 511872 | 507904 | 511872 | 511872 |
| `layout` | 0 | **−0.074** | 0 | 0 | 0 | 0 |
| `memory_plan` | 0 | 0 | 0 | 0 | 0 | 0 |

**这张表说的四件事**：

1. **`tiling` 是唯一在每个图上都有大杀伤的 pass**（访存 Δ 都在 50 万元素以上，
   最大的一格是 `attn` 的 761856）。
2. **`layout` 的延迟 Δ 是负数** ⇒ `blocked` 布局让延迟**变差**
   （占用率被 shared ×2 与寄存器 +4 压掉，而它买到的 shared 带宽 ×2
   在访存已经很少的时候不值）。**一个 pass 的 Δ 是负数，是合法产出**。
3. **`memory_plan` 在每一张图上访存 Δ 都是 0** —— 它只改峰值内存，不改访存。
4. **`cse` 在 `dual` 上是一对反号**：访存 Δ **+4096**（有用）、
   延迟 Δ **−2.817**（有害）。两个读数给出**相反的裁定**，
   而两个都是真的（见 B.5：它减计算、增核数）。

**逐项可加性（`is_additive`）**：11 个读数里只有 **3 个**可加
（`flops` / `n_kernels` / `peak_registers`），其余 8 个都有**交互残差**。
例：`dual` 的 `global_bytes` 残差 **−524288**（占总量 50%）。
⇒ 期望 `attribution_sums_to_total` 被**推翻**（B.7）。

## B.7 设计期望：**9 成立 / 6 被推翻**

| 期望 | 裁定 | 证据（从 `expectations.json` 重导出） |
| --- | --- | --- |
| `fusion_dominates_tiling` | ❌ **推翻** | `attn` fusion_gain 0.242 vs tiling_gain 0.702；`constbias` 0.009 vs 0.953。**分块全面胜过融合** |
| `attribution_sums_to_total` | ❌ **推翻** | 11 个读数里只有 3 个逐项可加；`dual` 的访存残差 −524288（占 50%） |
| `every_optional_pass_is_load_bearing` | ❌ **推翻** | `normalize`/`const_fold`/`cse`/`memory_plan` 在某些图上是**装饰** |
| `blocked_layout_improves_occupancy` | ❌ **推翻** | 5 张图一律 `blocked_better = 0`、`blocked_worse = 8` |
| `best_strategy_is_the_same_across_graphs` | ❌ **推翻** | 5 张图上 **2 条**不同的最优策略 |
| `registers_grow_with_tile_size` | ❌ **推翻** | `attn` [24,24,24,36] 与 `normmlp` [28,29,29,36] 在 t8/t16 上**不增** |
| `fusing_a_row_reduction_never_hurts` | ✅ 成立 | 5 图 × 32 对，`fusion_hurts_cells = 0` |
| `semantic_equivalence_is_bit_exact` | ✅ 成立 | 320/320，`max_abs_drift = 0.0` |
| `funnel_is_monotone` | ✅ 成立 | `violations = []` |
| `launch_count_equals_kernel_count` | ✅ 成立 | `violations = []` |
| `bigger_tile_reduces_traffic` | ✅ 成立 | 访存与延迟在 4 档 tiling 上**同时**单调 |
| `occupancy_never_exceeds_one` | ✅ 成立 | `violations = []` |
| `cse_never_increases_traffic` | ✅ 成立 | 5 图 × 64 格，`cells_where_cse_increases = 0` |
| `reuse_reduces_peak_memory` | ✅ 成立 | 5 图 × 64 格，`cells_where_reuse_is_worse = 0` |
| `distinct_strategies_give_distinct_cost` | ✅ 成立 | 64 格里 24～64 个**不同**的成本向量 |

**最值钱的三条被推翻**：

- **`fusion_dominates_tiling`** —— 写这条期望时想的是「融合是编译器的主菜」。
  实测：**融合省的是核数与 launch 开销，分块省的是访存**，
  而在不分块时矩阵乘的访存是 `O(MNK)`，融合省掉的 `O(MN)` 只是零头
  （比例低到 **1.009**）。**「融合比 tiling 值钱」是把两件不同的事放在一起比。**
- **`attribution_sums_to_total`** —— 消融是「一次只关一个」，
  而 `fusion × tiling` 有明显的交互（融合后的核才**值得**分块）。
  11 个读数里只有 3 个可加 ⇒ 「逐项可加」不是默认成立的性质，是要**验**的。
- **`blocked_layout_improves_occupancy`** —— 5 张图上一律更差（0 好 8 差）。
  这条期望曾经是**假设出来的**：布局轴的代价（shared ×2、寄存器 +4）
  写全了，收益（shared 带宽 ×2）却没写进模型 —— 于是「换布局一定更差」
  是**模型决定的**，不是**测出来的**。补上收益之后仍然是 0 好 8 差，
  这一次才是真的结论（C.333）。

## B.8 验收：**15 / 15**

| 门槛 | 阈值 | 实测 |
| --- | --- | --- |
| `min_graphs` | ≥ 4 | 5 |
| `min_passes` | ≥ 8 | 9 |
| `min_strategies` | ≥ 24 | 64 |
| `min_funnel_stages` | ≥ 8 | 10 |
| `min_kernels_per_graph` | ≥ 1 | 1 |
| `min_fusion_kernel_reduction` | ≥ 1.3× | **1.333×** |
| `min_tiling_gain_over_fusion` | ≥ 1.5× | **13.361×** |
| `min_total_traffic_reduction` | ≥ 2.0× | **14.980×** |
| `min_attributed_passes` | ≥ 3 | 7 |
| `min_lowering_paths_with_distinct_cost` | ≥ 2 | 24 |
| `min_cost_crosscheck_agreement` | ≥ 1.0 | 1.0 |
| `max_semantic_drift` | ≤ 0.0 | 0.0 |
| `min_cse_saved_ops` | ≥ 1 | 1 |
| `min_const_fold_saved_ops` | ≥ 1 | 1 |
| `min_normalize_saved_ops` | ≥ 1 | 1 |

⚠️ `min_fusion_kernel_reduction` 这条门槛**换过口径**：原来叫
`min_fusion_traffic_reduction`，盯的是访存 —— 而那个口径**量不出融合的功劳**
（实测 1.009×，一个真有杀伤的 pass 被判成装饰）。改成盯**核数降幅**（1.333×）
之后它才是有效的达标线（C.334）。

## B.9 对抗用例：**48 / 48**，且**证明它们能红**

| 项 | 值 |
| --- | --- |
| 用例数 | **48** |
| 成立 | **48** |
| 违反 | 0 |
| **能被证明能红** | **48 / 48** |
| ├ 被**通用变异**打红 | 38 |
| └ 需要**定向变异** | 10 |

工具：`.scratch/prove_cases_can_fail.py`。两段式：

1. **通用变异**：删键 / 改数 / 截列表 / 反转列表 / 删产物。
   只作用在该用例**真的读过**的产物上（用一个**探针 dict** 记下 `__getitem__` 的键）——
   不这么做的话通用变异池有 2 万条以上，每条用例只能试几十条，
   **打不中靶的概率极高，最后看起来像「用例恒真」，其实只是变异器没打到**（C.341）。
2. **定向变异**：通用变异打不动的 10 条，逐个给一条改 cfg / 改图 / 打源码补丁的变异，
   并**写清它对应哪种真缺陷**。

**10 条需要定向变异的**（都是「读的不是产物」的用例）：

| 用例 | 定向变异（= 它防的是哪种真缺陷） |
| --- | --- |
| `config_round_trips_through_disk` | 配置里塞一个代码不认识的口径键 ⇒ 磁盘配置读不回来 |
| `protocol_digest_is_stable` | 口径里塞一个集合 ⇒ 指纹不再由口径唯一决定 |
| `thresholds_are_all_wired_to_code` | `acceptance` 里加一个没人读的门槛键 |
| `every_expectation_has_an_evaluator` | 加一条没有求值函数的期望 |
| `tiles_divide_every_problem_dimension` | 把 `t32` 的 `tk` 改成 5 ⇒ tile 不整除 K |
| `every_graph_declares_at_least_one_output` | 把某张图的 `outputs` 清空 |
| `every_graph_has_a_matmul` | 把某张图的矩阵乘拿掉 |
| `every_graph_is_acyclic` | 把某张图改成有环 |
| `products_are_json_serializable` | 往产物里塞一个集合 |
| `no_elapsed_time_key_in_the_versioned_products` | 往进摘要的产物里塞一个实测耗时键 |

⚠️ **本轮靠这个工具抓出了 2 条「占着位置但什么都没验」的用例**（C.339 / C.340），
两条都**曾经是绿的**。这就是「恒真判据比没有判据更危险」的具体形状：
它们在报告里长得和别的用例一样。

## B.10 确定性核对与只读契约

| 核对 | 结果 |
| --- | --- |
| 两次 `run` 的 15 个产物 | **11 个逐位相同**；不同的 4 个：`version.json` / `benchmark.json` / `report.md` / `report.html`（理由见 A.4） |
| `verify` 的「意外不同」 | **（无）** |
| 产物充分性（从产物重建报告 == 直接渲染） | **True** |
| `visualize` | `report.md` **33325** 字节、`report.html` **53169** 字节，**逐位相同**（且与 `wc -c` 一致） |
| 只读命令跑完 `out/` | **逐位不变** |
| `--json` 的 stdout | **只含一段合法 JSON** |
| 从工作区根调用（路径锚定） | `check` / `show` / `visualize` / `verify`（含 `--json`）**rc = 0** |
| 锚定恒等性（6 个子命令 × 2 个路径参数） | **全部为恒等变换** |

⚠️ **`visualize` 比「渲染器是纯函数」更强**：它真的把产物**落盘再读回来**，
再重建报告。只比「同一份内存对象渲染两次」是**弱**断言 ——
两次读的是同一个 dict，连**插入序**都一样 ⇒ 渲染器偷看键序也照样绿（C.330）。

---

# 附录 C｜踩过的坑

> 编号跨项目连续。上一项目（`L6-kernelfactory`）止于 **323**，本附录从 **324** 开始。

### 324. `tile=None` 的语义写错 ⇒ 不分块反而访存**最少**，漏斗反向

**症状**：`tiling = "none"` 那一格（不分块）的访存**比 `t32` 还少**，
于是 `funnel_is_monotone_in_global_bytes` 红，而「分块降访存」这条直觉被反着证明。

**根因**：`tile=None` 被实现成「一整块装下整个问题 `(M, N, K)`」——
那样确实「一次读进来全部用完」，访存最少。但 `none` 的正确语义是
**逐输出元素读 `(1, 1, K)`**（含行归约时是 `(1, N, K)`，一行一行算）。

**修法**：`_tile_for_kernel` 里 `tile is None` 一律返回 `(1, 1, K)`
（行归约 `(1, N, K)`），并在 `tiling_table` 的注释里把「`none` 是 `null`，
不是 `[1,1,1]`」写死。

⚠️ 这个 bug 的危险之处：**它让一个不变量反向成立**。
`funnel_is_monotone` 红了才暴露它；如果只写「分块降访存」的期望，
它会以「期望被推翻」的形式出现，被当成「结论」而不是「bug」。

### 325. 循环上界有**两个来源** ⇒ 静态成本按 loops 乘、解释器按 tile 跑

**症状**：双成本互校在行归约的格子上不一致，而**两边各自的公式都没写错**。

**根因**：`_tile_for_kernel` 给出 `tn = 1`（行归约只看一列），
而 `build_kernel` 里**另写了一份特例**：行归约 ⇒ `j` 只走一次。
两个说法同时存在 ⇒ 静态成本按 `loops` 乘（少乘 N 倍），
解释器按 `tile` 跑（多跑 N 倍）。

**修法**：删掉特例，**循环一律由 tile 推出**（`ceil(dim / t)`），
并补一条判据 `kernel_loops_match_its_tile`：
`loops[:2] == (ceil(out_shape[0]/tm), ceil(out_shape[-1]/tn))`。

⚠️ 这条判据是**唯一**能直接钉住它的：没有它，这个 bug 只能靠
「双成本互校降了」间接发现，而它会伪装成「某个公式算错了」（C.324 同族）。

### 326. 广播的访存计数按**复制之后**算 ⇒ 双成本互校 236 条不一致

**症状**：320 格里 **236 条**不一致（一致率 26%）。

**根因**：`_tile_of` 在解释器里把行向量/列向量**预复制**成 `tm × tn`
再交给内核；动态计数于是按 `tm × tn` 记，而静态解析式按 `MemOp.elems`
（广播自己的尺寸）记。差 `tm` 或 `tn` 倍。

**修法**：`_tile_of` 返回**原始子块、不复制**；新增 `_as_tile`（把子块
广播成 `tm × tn` 供累加用）与 `_apply_binary`（就地广播）。
**计数用 raw，累加用广播** —— 这两件事必须分开。

⚠️ 广播的访存口径是「**这个 tile 迭代真的要从全局取的不同元素数**」：
读一次、寄存器里复用，不是「复制到 `tm × tn` 之后的元素数」。

### 327. IR 约定：**节点输出必须二维**（一维输出会被静默当成 `N×N`）

**症状**：`constbias` 图语义核对失败 8 格；更坏的是 —— 其中一部分**恰好也算出了正确结果**。

**根因**：IR 里一维中间张量（`[32]`）被当成 `[32, 32]` 参与形状推导。
在 `constbias` 上，因为广播的存在，某些格子**歪打正着**算出对的数。

**修法**：① 数据改成 `[1, 32]`；② `graph._validate_graph` **强制**
「任何节点输出必须二维」，把约定写进校验器而不是注释里。

⚠️ 「恰好也算对」是最难查的一种：它让 bug 只在一部分格子上显形，
看起来像「某个特定策略有问题」。

### 328. 常量折叠后的常量**必须有值** —— 值只能来自被折叠掉那个表达式的真值

**症状**：`KeyError: 'c'`（`constbias`）。

**根因**：`const_fold` 把 `c = b1 + b2` 折叠成一个常量叶子，
但**没人给它值**。更坏的做法是「用随机常量合成」——
那样程序跑完了，而语义核对红，看起来像「折叠算错了」。

**修法**：`pipeline.base_environment(g, cfg)` 在**原图上**跑一遍
`eval_graph`，把全部中间结果留下；阶梯与核对都从它取输入（`feeds_for`）。

⚠️ 一般化：**折叠掉一个表达式，就欠了它一个值**。
值必须来自「那个表达式真的算一遍」，不能来自「反正是个常量」。

### 329. 「工作量沿整条链恒定」是**错的**

**症状**：判据 `flops_never_increase_along_the_ladder` 原来写的是
「沿整条链**恒定**」，在 `dual` / `mlp3` 上红。

**根因**：`cse` 把 `dual` 里算了两遍的同一个 matmul 合并成一个 ⇒
工作量**真的减半**；`normalize` / `const_fold` 删掉算子 ⇒ 工作量也降。
「恒定」只在「flops 只数矩阵乘、且图级 pass 不动矩阵乘」时才成立，
而 CSE 恰恰会动矩阵乘。

**修法**：拆成两条 —— `flops_never_increase_along_the_ladder`（方向：不增）
+ `program_passes_do_not_change_flops`（**程序级 pass 一步都不改**）。
后者才是「改怎么算 / 改算什么」那条分界线。

⚠️ 「不增」是**弱**断言：融合时为了对齐而重算一点也能满足它。
逐级比**相等**才把它和「改算什么」分开。

### 330. 渲染器依赖**内存里的键序** ⇒ 从盘上重建不出同一份报告

**症状**：`visualize` 报「重建的报告与盘上的不同」，而
`renderers_are_pure_functions_of_the_products` 是**绿的**。

**根因**：渲染器遍历 `products["attribution.json"]["per_graph"]`（一个 dict）
时用了**插入序**。内存里的 dict 是插入序，而 `emit.dumps` 用
`sort_keys=True` 落盘 ⇒ 盘上读回来的键序是**字典序**，渲染出另一份文本。

**修法**：渲染器里所有遍历 dict 的地方加 `sorted(...)`；
判据**同时**比「同一份内存对象渲染两次」和「JSON 往返后的副本」——
只比前者是弱断言（两次读的是同一个 dict，连插入序都一样）。

⚠️ 「渲染器读了什么」本身就是一条不变量。`visualize` 比纯函数性更强，
因为它真的**落盘再读回来**。

### 331. `run` 与 `check` 对**同一批用例**给出不同裁定（一个环）

**症状**：`run` 说 47/48，`check` 说 48/48。两个数字说的是同一件事，却对不上。

**根因**：一个**环** —— 报告要印对抗用例的结论 ⇒ `render` 要读
`adversarial.json`；而对抗用例里有一条（`renderers_are_pure_functions_of_the_products`）
要调 `render`。第一趟跑的时候 `summary.json` 还没生成 ⇒ `render` 抛
`KeyError`，被 `except Exception` 兜成「违反」。

**修法**：这一层跑**两趟**。第一趟判「产物的性质」（排除
`REPORT_DEPENDENT_CASES`），产出的结论让 `summary.json` 与报告成型；
第二趟（`pipeline` 里显式调用）判「渲染器的性质」。
再补一条判据 `adversarial_cases_run_with_all_their_inputs_present`：
检查**第一趟跑的时候**入参有哪些键（`adversarial.json.inputs_present`），
不是「现在这一刻」—— 现在这一刻产物当然齐，拿它当证据等于没查。

⚠️ `except Exception` 兜底会把「缺产物」伪装成「违反」。
**兜底越宽，越需要一条断言去检查「判据是不是拿完整输入判的」。**

### 332. 一个**确定性**产物被错误地排除在版本摘要之外

**症状**：无（这正是它危险的地方）。

**根因**：`summary.json` 曾经被排除在 `VERSIONED_FILES` 之外，
理由写的是「引用了 `created_at` 与实测耗时」——
而它**两样都没有**（打开一看就知道）。

**修法**：加回 `VERSIONED_FILES`（现在 11 个），并把 `EXCLUDED` 里每一条的
**理由改成真的**（`report.md` 是「印了 `created_at`」，不是「嵌了耗时表」）。

⚠️ 排除清单里的理由**必须能被打开文件验证**。
一个「理由听起来很合理」的排除，会让一个确定性产物从此不进摘要 ——
而它照样在 `out/` 里，看起来一切正常。

### 333. 布局轴只写了**代价**没写**收益** ⇒ 「换布局一定更差」是假设出来的

**症状**：期望 `blocked_layout_improves_occupancy` 被推翻，5 张图一律
0 好 8 差 —— 但**这个结论不可信**。

**根因**：`blocked` 布局在成本模型里只登记了代价（shared ×2、寄存器 +4），
**收益（shared 带宽 ×2）没写**。于是「blocked 一定更差」是
**模型决定的**，不是**测出来的**。

**修法**：`LAYOUT_EFFECT` 改成三元组 `(shared 放大, 额外寄存器, shared 带宽倍数)`，
`blocked → (2.0, 4, 2.0)`，`kernel_latency_us` 里乘上 `shared_bandwidth_scale`。
补完收益之后**仍然是 0 好 8 差** —— 这一次才是真的结论。

⚠️ 一般化：**一个轴上的每个取值，代价与收益都要进模型**。
只写代价 ⇒ 那个轴是「假的」：它看起来有取舍，实际上模型已经替你选了。

### 334. 门槛盯错了量 ⇒ 一个真有杀伤的 pass 被判成装饰

**症状**：门槛 `min_fusion_traffic_reduction` 实测 **1.009×**，
几乎贴着 1.0 —— 看起来融合是个没用的 pass。

**根因**：不分块时矩阵乘的访存是 `O(MNK)`，融合省掉的是 `O(MN)` 那一小截。
**拿访存当融合的达标线，量的是另一个问题。**

**修法**：换成 `min_fusion_kernel_reduction`（**核数降幅**，实测 1.333×），
并把「为什么不能用访存」写进门槛的 `why` 字段里。

⚠️ 门槛的 `why` 文案是**被断言的一部分**：口径写错的门槛，
会在下一个项目里被原样抄走。

### 335. `_graph_pass_count` 取 `min` 不取 `max`

**症状**：`min_cse_saved_ops` / `min_const_fold_saved_ops` / `min_normalize_saved_ops`
三条门槛红，而 `dual` 上 CSE 明明省了算子。

**根因**：取 `min` 等于问「**每张图**都必须有活干」，而这三条门槛问的是
「**这一级**有没有活干」。

**修法**：改取 `max`。

### 336. 最优延迟点**不在前沿上** ⇒ tie-break 只用了**一根**目标轴

**症状**：`best_latency_point_is_on_the_pareto_front` 在 `attn` / `dual` / `normmlp` 上红。

**根因**：`min(cells, key=lambda c: (c["readings"]["latency_us"], c["strategy_id"]))`
在有并列最优时，用 `strategy_id` 破并列 —— 而 `strategy_id` 与**支配关系无关**，
于是选出一个被别的格子支配的点。

**修法**：`_best(cells, axis)`：tie-break 用**全部三个目标轴**，最后才用 id。

⚠️ 「按 id 破并列」看起来无害（「反正并列，选哪个都一样」），
但在**多目标**下并列只在一个轴成立，另外两个轴可能一好一坏。

### 337. 分块覆盖比对错了**对象**：拿原图的算子名去比这一级的图

**症状**：`partition_covers_every_node_exactly_once` 在 **160 格**上红。

**根因**：图级 pass（normalize / const_fold / cse）会**删算子**，
而判据拿 `data/graphs/*.json` 里那张**原图**的算子名去比这一格的 `partition`。

**修法**：`compare.cells` 里加 `final_nodes`（这一级图的算子名），判据比它。

⚠️ 判据必须与声明指**同一个对象**：声明说的是「分块覆盖了**这一级图**的
全部算子」，不是「覆盖了原图」。

### 338. 「实测耗时」与「模型预测」混住

**症状**：`no_elapsed_time_key_in_the_versioned_products` 红，而 `latency_us`
是**模型预测**（`Σ(launch + 访存 + 算力)`），不是墙钟测量。

**根因**：判据按**键名正则**抓时间类键，`latency_us` / `launch_overhead_us`
命中。而它们确实是**成本模型的确定性函数**（同一份口径必得同一个值）。

**修法**：加一个**短白名单** `_MODEL_TIME_KEYS = {latency_us, launch_overhead_us}`，
断言「**命中集合 ⊆ 白名单**」；豁免标志 `exempt` 必须**沿子树传下去**
（白名单键下面可能是一个**列表** `"latency_us": [10.6, 9.5, ...]`，
在标量层回头看 `path[-1]` 是 `"[0]"`，看不出来源）。

⚠️ 白名单是**白名单**不是黑名单：收的是**全部**命中正则的键，
然后断言它 ⊆ 白名单。新增一个时间类键名必须**显式报到**。
白名单一旦开始变长，就说明两类东西在产物里混住了。

### 339. **恒真判据**：`deep_merge` 是幂等的 ⇒ 「往返核对」什么都没做

**症状**：`config_round_trips_through_disk` 一直是绿的。它**从来不可能是红的**。

**根因**：它比的是 `deep_merge(DEFAULTS, c)` 与
`deep_merge(DEFAULTS, deep_merge(DEFAULTS, c))` —— 而 `deep_merge`
**是幂等的**，两个表达式**恒等**。名字里的「through disk」一次都没发生过：
**它没有碰过磁盘**。

**修法**：真的往返 —— `dump_config(clean, 临时文件)` → `load_config(临时文件)`
→ 两份逐位比。顺带把「保险丝会不会误伤自己导出的配置」也验了。

⚠️ 这条是被**变异测试**抓出来的：48 条用例逐条找「能让它红的输入」，
只有它和 C.340 两条**找不到**。**恒真的判据比没有判据更危险**，
因为它占着一个「这里验过了」的位置，而报告里长得和别的判据一样。

### 340. **空转判据**：参考链上**一条 `skipped` 级都没有**

**症状**：`skipped_stages_do_not_change_the_program` 一直是绿的。

**根因**：`build_ladder` 用「**全开**」策略建参考链 ⇒
`lower_ladder` 里 `enabled` 恒为真 ⇒ `skipped` 恒为假。
那条断言的核心循环 `if r["skipped"] and i > 0:` **一次都没执行过**，
`bad` 恒为空。

**修法**：建**第二条**降级链（`baseline_strategy` = 布尔轴全 `False`、
枚举轴取各自第一项），里面有 **10 个真的 `skipped` 级**；
判据改成读**两条链**，并且要求 `n_skipped > 0` ——
否则**红**，而不是绿着假装验过。

⚠️ 「这条断言在参考数据上有没有东西可判」是一个**必须单独问的问题**。
两条（C.339 / C.340）都是靠变异测试问出来的。

### 341. 变异测试的**靶**：通用变异必须只打在用例**真的读过**的产物上

**症状**：第一版变异脚本报「46/48 条用例打不动」，看起来像「用例恒真」。

**根因**：通用变异池有 **26543** 条，而每条用例只试前 40 条 ⇒
**打不中靶的概率极高**。而且脚本自己还有个 bug：
「删产物」那一类变异的参数是 `{name: None}`，`p.pop(dict)` 直接
`TypeError: unhashable type: 'dict'`。

**修法**：用一个**探针 dict**（重写 `__getitem__` / `get` / `__contains__` /
`items` / `keys` / `__iter__` 记下访问过的键）先跑一遍用例，
把变异池**限定在它读过的产物上**；并且内容变异排在「删产物」之前
（删产物只能证明「用例读了这份产物」，证明不了「它在看里面的**值**」）。

⚠️ 「用例恒真」与「变异器打不中靶」**必须分得清**。
分不清的话，你会去改一个本来没问题的判据。

### 342. `skipped` 说的是「pass 有没有被**跑**」，不是「它有没有**改变**东西」

**症状**：加完基线链之后，`tiling` / `layout` / `memory_plan` 三个 pass
在「全关」策略下 `skipped = False`，而它们**什么都没改**。

**根因**：`enabled = name not in optional or bool(strat.get(name, True))`。
`tiling = "none"` 是**真值字符串** ⇒ `enabled = True` ⇒ pass **跑了**、
决定不分块。只有布尔轴取 `False` 才是真的没跑。

**修法**：不改语义（现在的语义是对的），但把它**写进注释与报告**：
`skipped` 是「跑没跑」；`tiling=none` 的 pass 跑了 ⇒ `skipped=False`。
两者混起来的话，「跳过级读数不变」会被误读成「不分块那一级也该不变」。

### 343. `argparse.SUPPRESS` 吞掉默认值 ⇒ `Namespace` 上没有那个属性

**症状**：`TypeError: 'Namespace' object has no attribute 'json'`。

**根因**：子命令的 `--json` / `--config` / `--out` 用了
`default=argparse.SUPPRESS`（为了让「子命令没给就用父命令的」），
于是用户没给时属性**根本不存在**，`args.json` 直接抛。

**修法**：`cli._fill_defaults(args)` 在 `parse_args` 之后把缺的属性补回来，
**再**调 `_anchor_args`。

### 344. 生成器不能当 context manager

**症状**：`TypeError: 'generator' object does not support the context manager protocol`。

**根因**：`_quiet_prose` 写成了裸生成器函数（`yield` 一次），
用 `with _quiet_prose():` 调用 ⇒ 拿到的是 generator，不是 CM。

**修法**：加 `@contextlib.contextmanager`。

### 345. 一个叫 `bytes` 的字段，量的是**字符数**；而盘上的文件还多一层 CRLF

**症状**：`visualize` 说「`report.md` **25161 字节**」，`wc -c` 说 **33784**。
两个数都对，但它们**不是同一个东西** —— 而报告里两处都印着「字节」。

**根因**两层，叠在一起：

1. `len(text)` 是**字符数**。报告里全是中文（UTF-8 下一个字 3 字节）
   ⇒ 字符数与字节数差 1.3 倍以上。字段却叫 `bytes`，命令里也印「字节」。
2. 就算改成 `len(text.encode("utf-8"))`，还是对不上：
   `Path.write_text` 在 Windows 上会把 `\n` 翻译成 `\r\n`
   ⇒ 盘上文件比渲染出来的文本**多「行数」那么多字节**。

**修法**：① `visualize` / `verify` 的 `bytes` 改成真的
`len(text.encode("utf-8"))`（另存一个 `chars`）；
② 所有写盘点加 `newline="\n"`，让盘上的文件与渲染文本**逐字节相同**。

⚠️ 一般化：**「字节」这个词在跨平台代码里必须钉死是哪一层**——
`str` 的长度、`encode()` 的长度、和盘上文件的长度，是**三个**不同的数。
一个不区分它们的核对，会在两个平台上给出两个都「通过」的结论。

### 346. **CSE 减少计算，同时制造扇出、破坏融合** ⇒ 延迟反而变差 45%

**症状**：`dual` 图上 `cse=on` 比 `cse=off` **慢**（9.211 µs vs 6.357 µs），
而它的工作量是**一半**（134144 vs 265216）。

**根因**：`fuse` 的三条判据之一是「**中间张量只被用一次**」。
`cse` 把两个 `matmul(A, B)` 合并成一个 `t0` 之后，`t0` 被**用了两次**
（一次走 `gelu`、一次走 `add`）⇒ 融合被拆开 ⇒
核数 `2 → 3`，而 `launch` 开销（每次 3 µs）**主导**延迟。

**修法**：不改。这是一个**真结论**，要报出来：

- 归因表里 `cse` 在 `dual` 上访存 Δ **+4096**、延迟 Δ **−2.817** ——
  两个读数给出**相反**的裁定，而两个都是真的。
- 期望 `best_strategy_is_the_same_across_graphs` 因此被推翻
  （最优策略在 5 张图上不唯一）。

⚠️ 一般化：**「少干活」不等于「更快」**。当固定开销（这里是每次 launch 3 µs）
主导时，一个减少工作量的 pass 完全可能让目标变差 ——
而只看 `flops` 那一条轴会得出**相反**的结论。
多目标的意义就在这里：**一条轴上的改善，必须在另一条轴上复核。**

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`；
附录 A/B/C 为落地后补写，其中附录 B 的每一个数字都从最终 `out/` 重新导出。_
