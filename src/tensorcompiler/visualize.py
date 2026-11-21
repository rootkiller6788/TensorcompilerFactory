"""只读重建：从 `out/` 的产物重建报告，并与盘上的报告**逐位比对**。

这条命令的命题是「**产物足以重造报告**」—— 它不重跑流水线，
只做 `read_products → render → 比对`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .emit import read_products
from .report import render


def rebuild(out_dir: str | Path) -> tuple[dict[str, Any], str, str]:
    products = read_products(Path(out_dir))
    md, html = render(products)
    return products, md, html


def verify(out_dir: str | Path) -> dict[str, Any]:
    """⚠️ `bytes` 必须是**真的字节数**（`len(text.encode("utf-8"))`），
    不是 `len(text)` —— 后者是**字符数**，而报告里全是中文。

    这里踩过一次：字段叫 `bytes`、命令里也印「字节」，量的却是字符。
    于是「`visualize` 说 25161 字节」与 `wc -c` 的 33784 **对不上**，
    而两边都没错 —— 它们量的不是同一个东西（C.345）。
    """
    out = Path(out_dir)
    products, md, html = rebuild(out)
    rows = []
    for name, text in (("report.md", md), ("report.html", html)):
        p = out / name
        row = {"file": name, "chars": len(text),
               "bytes": len(text.encode("utf-8"))}
        if not p.exists():
            rows.append({**row, "exists": False, "bitwise_equal": False})
            continue
        disk = p.read_text(encoding="utf-8")
        rows.append({**row, "exists": True, "bitwise_equal": disk == text})
    return {"out_dir": str(out), "rows": rows,
            "all_equal": all(r["bitwise_equal"] for r in rows)}
