"""产物写入与**内容寻址版本**。

⚠️ **版本摘要只覆盖确定性产物**（`VERSIONED_FILES`），显式排除：
① `version.json`（自指 + 时间戳）；
② **耗时类产物**（随机器变）—— 本项目里是 `benchmark.json`；
③ `report.md` / `report.html`（**视图**，正文里嵌了耗时表）。

⇒ 核对要精确表述成「两次 `run` 只有 `version.json` 与 `<耗时文件>` 不同，
其余 N 个逐位相同」，不是「产物完全一样」。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: 参与版本摘要的产物（**确定性**的结论性产物）。
VERSIONED_FILES: tuple[str, ...] = (
    "graphs.json",
    "ladder.json",
    "compare.json",
    "attribution.json",
    "semantics.json",
    "crosscheck.json",
    "acceptance.json",
    "adversarial.json",
    "expectations.json",
    "config.json",
    # ⚠️ `summary.json` 曾经被排除，理由是「引用了 created_at 与实测耗时」——
    #    而它**两样都没有**。一个被错误排除的确定性产物，代价是版本摘要
    #    覆盖不到头条数字：改了 summary 的算法，版本号**不会变**。
    "summary.json",
)

#: 显式排除在版本摘要之外的文件，**每个都要有理由**。
#: ⚠️ 理由必须写**真实的**那一句。写「引用了实测耗时」而实际是
#: 「引用了 created_at」，会让下一个想把它加回摘要的人被误导 ——
#: 注释与代码说的不是同一件事时，读者会信注释。
EXCLUDED: dict[str, str] = {
    "version.json": "自指（它的摘要覆盖别的东西）+ created_at 时间戳",
    "benchmark.json": "**实测**耗时 —— 每次运行都不同，是知情的取舍不是 bug",
    "report.md": "视图：正文里印了 created_at（每次 run 都不同）",
    "report.html": "视图：同 report.md",
}


def dumps(obj: Any) -> str:
    """**统一序列化**：`sort_keys=True` + 固定缩进。

    ⚠️ 忘写 `sort_keys=True` 会让 set 迭代序泄漏进产物 —— 症状是
    「两次 run 的产物不同」，而代码看起来完全确定。
    """
    return json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def write_product(out_dir: Path, name: str, obj: Any) -> Path:
    p = Path(out_dir) / name
    # ⚠️ `newline="\n"`：`write_text` 在 Windows 上会把 `\n` 翻成 `\r\n`，
    #    于是**盘上文件的字节数**比渲染出来的文本多「行数」那么多字节。
    #    症状是「`visualize` 说 33325 字节」与 `wc -c` 的 33784 对不上，
    #    而两边都没错 —— 它们量的不是同一个东西（C.345）。
    p.write_text(dumps(obj), encoding="utf-8", newline="\n")
    return p


def digest_of(products: dict[str, Any]) -> str:
    """内容寻址：只吃 `VERSIONED_FILES`。"""
    h = hashlib.sha256()
    for name in VERSIONED_FILES:
        if name not in products:
            raise KeyError(f"版本摘要要求 {name}，但产物里没有 —— "
                           "`VERSIONED_FILES` 与实际产物必须一致")
        h.update(name.encode("utf-8"))
        h.update(dumps(products[name]).encode("utf-8"))
    return h.hexdigest()[:12]


def read_products(out_dir: Path) -> dict[str, Any]:
    """把 `out/*.json` 读成产物字典（渲染器**只吃这个**）。"""
    out: dict[str, Any] = {}
    for f in sorted(Path(out_dir).glob("*.json")):
        out[f.name] = json.loads(f.read_text(encoding="utf-8"))
    return out
