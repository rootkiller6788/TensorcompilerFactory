"""配置：口径与门槛。

**代码里不硬编码阈值** —— 所有门槛、成本模型常数、策略空间都在这里，
磁盘上的 `configs/default.json` 是它的落盘副本。

⚠️ `deep_merge` 里**列表整体替换、磁盘优先** ⇒ 改了 `DEFAULTS` 里的列表
（比如 `passes.optional`、`expectations`）而没重新导出，磁盘上的旧列表会
**静默盖掉**新声明（「改了声明、重跑，数字一字未变」）。
⇒ 改完默认值**必须**跑 `python main.py dump-config`。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------- 默认配置
DEFAULTS: dict[str, Any] = {
    "seed": 20261002,
    "graphs_dir": "data/graphs",

    # ---- pass 链 -------------------------------------------------------
    #: `order` 是**全部** pass 的合法顺序（拓扑序）；`optional` 里的可以被策略关掉。
    #: ⚠️ 这两条都是**列表** ⇒ 改完必须 dump-config，否则磁盘旧列表会盖掉新声明。
    "passes": {
        "order": [
            "normalize",      # 代数简化：kill 恒等算子
            "shape_infer",    # 形状推导
            "const_fold",     # 常量子图折叠
            "cse",            # 公共子表达式消除
            "fusion",         # producer→consumer 融合（决定 kernel 数）
            "tiling",         # 循环分块（决定访存）
            "layout",         # 布局（决定 shared / registers）
            "memory_plan",    # buffer 复用（决定峰值内存）
            "lower",          # 落到可执行的 Kernel IR（决定线程映射）
        ],
        "optional": ["normalize", "const_fold", "cse", "fusion",
                     "tiling", "layout", "memory_plan"],
    },

    # ---- 策略空间（交叉积展开）-----------------------------------------
    "strategies": {
        "fusion": [True, False],
        "cse": [True, False],
        "tiling": ["none", "t8", "t16", "t32"],
        "layout": ["row_major", "blocked"],
        "memory_plan": ["naive", "reuse"],
    },
    #: 每一档 tiling 的名字 → matmul 的 (tm, tn, tk)。
    #: ⚠️ `none` 是 **`null`，不是 `[1,1,1]`** —— 「不分块」不是「1×1×1 的分块」：
    #: 前者的循环嵌套是 `[i(M), j(N)]` + 内层整条 K 归约（逐输出元素读），
    #: 后者是 `[i(M), j(N), k(K)]` 每格读一个元素。两者访存差 O(K) 倍。
    "tiling_table": {
        "none": None,
        "t8": [8, 8, 8],
        "t16": [16, 16, 16],
        "t32": [32, 32, 32],
    },

    # ---- 成本模型（**全部是模型假设，不是实测**）------------------------
    #: ⚠️ 这张表是「口径」，报告里必须显式标注哪些读数是算出来的。
    "cost": {
        "bytes_per_elem": 4,               # fp32
        "launch_overhead_us": 3.0,         # 每次 kernel launch 的固定开销
        "global_bandwidth_gb_s": 600.0,    # 全局访存带宽
        "shared_bandwidth_gb_s": 4000.0,   # 共享内存带宽
        "flops_per_s": 1.0e12,             # 算力
        # ---- occupancy 模型 ----
        "sm_count": 20,
        "register_file_per_sm": 65536,
        "shared_per_sm": 49152,
        "max_threads_per_sm": 1024,
        "threads_per_block": 256,
    },

    # ---- 验收门槛（**可调达标线**）-------------------------------------
    "acceptance": {
        "min_graphs": 4,
        "min_passes": 8,
        "min_strategies": 24,
        "min_funnel_stages": 8,
        "min_kernels_per_graph": 1,
        "min_fusion_kernel_reduction": 1.3,
        "min_tiling_gain_over_fusion": 1.5,
        "min_total_traffic_reduction": 2.0,
        "min_attributed_passes": 3,
        "min_lowering_paths_with_distinct_cost": 2,
        "min_cost_crosscheck_agreement": 1.0,
        "max_semantic_drift": 0.0,
        "min_cse_saved_ops": 1,
        "min_const_fold_saved_ops": 1,
        "min_normalize_saved_ops": 1,
    },

    # ---- 设计期望（**允许被实测推翻**）---------------------------------
    #: 每条：`id` + `claim`（声明）+ `detail`（口径）。裁定在 `expectations.py`。
    #: ⚠️ **不追求全绿** —— 全绿说明期望写得太保守（等于没写）。
    #: ⚠️ 这几条是**先写下来再去量的**，不是量完回头补的。
    "expectations": [
        {"id": "fusion_dominates_tiling",
         "claim": "融合带来的访存降幅 > 分块带来的访存降幅",
         "detail": "两者都按「相对朴素基线的访存字节降幅」算，同一口径"},
        {"id": "attribution_sums_to_total",
         "claim": "各 pass 的消融贡献之和 == 全链相对朴素基线的总变化",
         "detail": "消融是「一次只关一个」，逐项可加需要**没有交互项**"},
        {"id": "every_optional_pass_is_load_bearing",
         "claim": "每一个可选 pass 在每一张图上被关掉后，至少有一个读数变化",
         "detail": "否则它是装饰 —— 但「装饰」本身是合法的产出，只是要报出来"},
        {"id": "blocked_layout_improves_occupancy",
         "claim": "`blocked` 布局的占用率不低于 `row_major`",
         "detail": "同一格其他轴固定，只换 layout 比较占用率"},
        {"id": "fusing_a_row_reduction_never_hurts",
         "claim": "把行归约融进矩阵乘，访存不会变差",
         "detail": "同一格其他轴固定，只换 fusion 开关比较访存"},
        {"id": "best_strategy_is_the_same_across_graphs",
         "claim": "延迟最低的那条策略在**所有图**上是同一条",
         "detail": "「同一条」= strategy_id 相同（strategy_id 只由轴值决定）"},
        {"id": "semantic_equivalence_is_bit_exact",
         "claim": "降级前后**逐位相等**（不是「近似」）",
         "detail": "浮点加法不满足结合律，所以这条要求降级不改变归约顺序"},
        {"id": "funnel_is_monotone",
         "claim": "沿 pass 链，全局访存逐级不增",
         "detail": "没有哪个 pass 会让程序做更多全局访存"},
        {"id": "launch_count_equals_kernel_count",
         "claim": "kernel launch 数恰好等于核数",
         "detail": "launch 是核的属性，不是运行期的"},
        {"id": "bigger_tile_reduces_traffic",
         "claim": "分块越大，访存越少**且延迟越低**",
         "detail": "同时看访存与延迟两个读数 —— 只报访存会漏掉代价"},
        {"id": "registers_grow_with_tile_size",
         "claim": "分块越大，寄存器压力越大",
         "detail": "更大的 tile 要更多的累加器与索引"},
        {"id": "occupancy_never_exceeds_one",
         "claim": "occupancy ≤ 1.0",
         "detail": "占用率是比例；>1 说明模型算错了"},
        {"id": "cse_never_increases_traffic",
         "claim": "开 CSE 后的全局访存 ≤ 关 CSE",
         "detail": "CSE 只删重复计算，不改变剩余算子的访存"},
        {"id": "reuse_reduces_peak_memory",
         "claim": "buffer 复用的峰值内存 ≤ 不复用",
         "detail": "复用是「同一块显存干两件事」，不会让峰值变大"},
        {"id": "distinct_strategies_give_distinct_cost",
         "claim": "不同的降级策略给出不同的成本向量",
         "detail": "否则「多策略对比」没有信息量"},
    ],
}


class ConfigError(Exception):
    """配置不合法。**一律抛错**，不静默回落（附录 C：宽松的 load_config
    会把「路径错了」伪装成「用默认值」）。"""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并；**列表整体替换**（这是「磁盘旧列表盖掉新声明」的来源，
    所以有 `no_stale_config_keys` + 必须 dump-config 的纪律）。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, *,
                root: str | Path | None = None) -> dict[str, Any]:
    """加载配置。`root` **真的参与路径解析**。"""
    if path is None:
        base = Path(root) if root is not None else Path.cwd()
        p = base / "configs" / "default.json"
    else:
        p = Path(path)
        if not p.is_absolute() and root is not None:
            p = Path(root) / p
    if not p.exists():
        raise ConfigError(f"配置文件不存在：{p}")
    try:
        disk = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件不是合法 JSON：{p} —— {e}") from e
    cfg = deep_merge(DEFAULTS, disk)
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(Path(root) if root is not None else Path.cwd())

    stale = no_stale_config_keys(disk, DEFAULTS)
    if stale:
        raise ConfigError(
            "配置里有代码中不存在的键（多半是从 DEFAULTS 删了键但磁盘配置没跟着更新）：\n"
            + "\n".join(f"  - {k}" for k in stale)
            + f"\n  配置文件：{p}\n"
            "  修法：用 `tensorcompiler.config.dump_config(DEFAULTS, path)` 重新导出。"
        )
    validate_config(cfg)
    return cfg


def no_stale_config_keys(disk: dict[str, Any], defaults: dict[str, Any],
                         prefix: str = "") -> list[str]:
    """磁盘上出现了代码里没有的键 ⇒ 报出来（保险丝 `no_stale_config_keys`）。"""
    stale: list[str] = []
    for k, v in disk.items():
        here = f"{prefix}{k}"
        if k not in defaults:
            stale.append(here)
        elif isinstance(v, dict) and isinstance(defaults[k], dict):
            stale.extend(no_stale_config_keys(v, defaults[k], here + "."))
    return stale


def validate_config(cfg: dict[str, Any]) -> None:
    """配置校验。**每条报错写清「为什么这是错的」** —— 文案本身是被断言的一部分。"""
    errs: list[str] = []

    if int(cfg["seed"]) <= 0:
        errs.append("seed 必须 > 0（0 不是种子，是「没设种子」）")

    # ---- pass 链 ----
    order = list(cfg["passes"]["order"])
    optional = list(cfg["passes"]["optional"])
    if len(set(order)) != len(order):
        errs.append(f"passes.order 里有重复的 pass：{order}")
    unknown = [p for p in optional if p not in order]
    if unknown:
        errs.append(f"passes.optional 里的 pass 不在 passes.order 里：{unknown}"
                    "（一个永远不会被跑到的开关，与没有这个开关在输出里一样）")
    if len(order) < int(cfg["acceptance"]["min_passes"]):
        errs.append(f"passes.order 只有 {len(order)} 个，门槛要求 ≥ "
                    f"{cfg['acceptance']['min_passes']}")

    # ---- 策略空间 ----
    strat = cfg["strategies"]
    if not strat:
        errs.append("strategies 不能为空 —— 空空间 ⇒ 矩阵缺口 = 0 是平凡满足")
    for k, v in strat.items():
        if not isinstance(v, list) or not v:
            errs.append(f"strategies.{k} 必须是非空列表，实际 {v!r}"
                        "（空轴会让「矩阵缺口 = 0」变成平凡满足）")
    n = 1
    for v in strat.values():
        n *= len(v)
    if n < int(cfg["acceptance"]["min_strategies"]):
        errs.append(f"策略空间只有 {n} 条，门槛要求 ≥ {cfg['acceptance']['min_strategies']}")
    for name in strat.get("tiling", []):
        if name not in cfg["tiling_table"]:
            errs.append(f"strategies.tiling 里的 {name!r} 在 tiling_table 里没有定义")
        else:
            raw = cfg["tiling_table"][name]
            if raw is not None and len(raw) != 3:
                errs.append(f"tiling_table[{name!r}] 必须是三元组 (tm,tn,tk) 或 null，"
                            f"实际 {raw!r}")

    # ---- 成本模型 ----
    c = cfg["cost"]
    for k in ("bytes_per_elem", "launch_overhead_us", "global_bandwidth_gb_s",
              "shared_bandwidth_gb_s", "flops_per_s", "sm_count",
              "register_file_per_sm", "shared_per_sm", "max_threads_per_sm",
              "threads_per_block"):
        if k not in c:
            errs.append(f"cost.{k} 缺失 —— 成本模型必须**显式**列出每一个假设")
        elif float(c[k]) <= 0:
            errs.append(f"cost.{k} 必须 > 0，实际 {c[k]!r}")
    if "threads_per_block" in c and "max_threads_per_sm" in c:
        if int(c["threads_per_block"]) > int(c["max_threads_per_sm"]):
            errs.append("cost.threads_per_block > max_threads_per_sm —— 一个 block 装不下")

    # ---- 门槛 ----
    for k, v in cfg["acceptance"].items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            errs.append(f"acceptance.{k} 必须是数字，实际 {v!r}")
        elif v < 0:
            errs.append(f"acceptance.{k} 不能是负数，实际 {v!r}")

    # ---- 期望 ----
    ids = [e["id"] for e in cfg["expectations"]]
    if len(set(ids)) != len(ids):
        errs.append(f"expectations 里有重复的 id：{ids}")
    for e in cfg["expectations"]:
        if not e.get("claim"):
            errs.append(f"期望 {e['id']} 没有 claim —— 没有声明的期望无法被推翻")
        if not e.get("detail"):
            errs.append(f"期望 {e['id']} 没有 detail —— 口径没写清就无法判定")

    if errs:
        raise ConfigError("配置校验失败：\n" + "\n".join(f"  - {e}" for e in errs))


def protocol_digest(cfg: dict[str, Any]) -> str:
    """协议指纹：只吃**口径**（pass 链 / 策略空间 / 成本模型 / 门槛），不吃路径与时间。"""
    payload = {
        "passes": cfg["passes"],
        "strategies": cfg["strategies"],
        "tiling_table": cfg["tiling_table"],
        "cost": cfg["cost"],
        "acceptance": cfg["acceptance"],
        "expectations": cfg["expectations"],
    }
    import hashlib
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    """把当前生效的配置写回磁盘。

    ⚠️ **这会写盘**。存在的理由是：磁盘配置是 `DEFAULTS` 的副本，而 `deep_merge`
    里**列表整体替换、磁盘优先** ⇒ 磁盘上的旧列表会让代码里的新声明静默失效。
    """
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    Path(path).write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
