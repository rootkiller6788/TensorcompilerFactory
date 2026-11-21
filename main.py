"""入口：`python main.py <cmd>`。逻辑全在 `src/tensorcompiler/cli.py`。"""

from __future__ import annotations

import sys
from pathlib import Path

# ⚠️ Windows 上 stdout 一旦被**重定向或接管道**，编码就退回 locale（常见是 GBK），
#    而 `--help` 文案里有 `⚠️` 这类 GBK 编不出的字符 ⇒ `UnicodeEncodeError`、
#    进程以 rc=1 退出。终端里看不出来（终端能编码的字符恰好够用），
#    只有「重定向 / 子进程捕获输出」这条路径会崩 —— 测试正是这么调 CLI 的。
#    ⇒ 入口处把两个流显式钉到 UTF-8。`reconfigure` 是唯一不改写调用方的做法；
#    在终端、管道、文件重定向下都给出同一串字节。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensorcompiler.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
