"""本轮优化/加固的回归测试。

⚠️ 这些不是第二套判据（判据的真源仍是流水线自身的门槛 / 对抗用例 / 期望），
它们是**入口闸**：确保本轮修掉的几类问题不会悄悄回来 ——
输出的编码、报告里「字节」的口径、拓扑序的形状、被跳过的级不该重建、
以及配置/图写错时要在加载期就报错。

⚠️ 自包含（不依赖 conftest 夹具），与同目录另外两个测试文件一致。
"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorcompiler import report as R  # noqa: E402
from tensorcompiler.config import (  # noqa: E402
    DEFAULTS, ConfigError, load_config, validate_config,
)
from tensorcompiler.graph import _validate_graph, load_graphs  # noqa: E402
from tensorcompiler.ir import Graph, Node, Tensor  # noqa: E402
from tensorcompiler.passes import lower_ladder  # noqa: E402
from tensorcompiler.compare import baseline_strategy  # noqa: E402


# --------------------------------------------------------------------- CLI
def test_help_is_utf8_when_stdout_is_a_pipe() -> None:
    """Windows 上被重定向/接管道时，--help 不能因 locale 编码崩掉。"""
    r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--help"],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="strict")
    assert r.returncode == 0, r.stderr
    assert "run" in r.stdout and "dump-config" in r.stdout


# --------------------------------------------------------------------- 报告
def test_report_footer_measures_bytes_not_chars() -> None:
    """列名是「字节」就必须是字节：中文产物的字符数与字节数不同。"""
    p = {
        "a.json": {"x": "中文"},
        "version.json": {"version": "v1", "created_at": None,
                         "versioned_files": ["a.json"]},
    }
    text = "\n".join(R._section_footer(p))
    expected = len(R._dump(p["a.json"]).encode("utf-8"))
    assert expected != len(R._dump(p["a.json"]))  # 否则这条测不出区别
    assert f"| `a.json` | {expected} |" in text


# --------------------------------------------------------------------- 拓扑序
def test_topo_order_is_level_batched_and_cached() -> None:
    """批内按名字排、整批取出、再进下一批 —— 不是全局按名字排。

    全局排序会把 `a2`（依赖 `z`）排到 `m` 前面；分批 Kahn 不会。
    """
    g = Graph(
        "diamond", "",
        (Tensor("a", (2, 2)), Tensor("b", (2, 2)),
         Tensor("z", (2, 2)), Tensor("m", (2, 2)), Tensor("a2", (2, 2))),
        (Node("gelu", ("a",), "z"),
         Node("relu", ("b",), "m"),
         Node("gelu", ("z",), "a2")),
        ("a2", "m"),
    )
    got = [n.output for n in g.topo_order()]
    assert got == ["m", "z", "a2"]
    assert g.topo_order() is g.topo_order()   # 缓存复用同一个 tuple


def test_topo_order_rejects_duplicate_outputs() -> None:
    g = Graph(
        "dup", "",
        (Tensor("a", (2, 2)), Tensor("c", (2, 2))),
        (Node("gelu", ("a",), "c"), Node("relu", ("a",), "c")),
        ("c",),
    )
    with pytest.raises(ValueError, match="重复的节点输出"):
        g.topo_order()


# --------------------------------------------------------------------- 阶梯
def test_skipped_stages_reuse_the_previous_program() -> None:
    """基线链（全部可选 pass 关闭）里，每个跳过级都不该重新 build。"""
    cfg = load_config(ROOT / "configs" / "default.json", root=ROOT)
    g = load_graphs(cfg)[0]
    stages = lower_ladder(g, baseline_strategy(cfg), cfg)
    assert len(stages) > 1
    first = stages[0].program
    for st in stages[1:]:
        if st.skipped:
            assert st.program is first


# --------------------------------------------------------------------- 校验
def test_validation_rejects_duplicate_strategy_values() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["strategies"]["fusion"] = [True, True]
    with pytest.raises(ConfigError, match="重复取值"):
        validate_config(cfg)


def test_validation_rejects_unknown_strategy_axis() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["strategies"]["fussion"] = [True, False]   # 拼错
    with pytest.raises(ConfigError, match="不在 passes.optional"):
        validate_config(cfg)


def test_graph_validation_rejects_3d_leaf() -> None:
    g = Graph(
        "bad3d", "",
        (Tensor("a", (2, 3, 4)), Tensor("c", (2, 4))),
        (Node("transpose", ("a",), "c"),),
        ("c",),
    )
    with pytest.raises(ValueError, match="3 维"):
        _validate_graph(g, Path("bad3d.json"))
