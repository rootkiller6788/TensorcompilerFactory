"""CLI：`python main.py <cmd>`。

⚠️ 每个子命令的 `description` 里**必须恰好出现一个**显式标记：
`会写盘` 或 `只读`。README 附录 A.3 的「写盘」列用的是**同一套标记**，
`.scratch/check_readme.py` 按这两个词逐命令机器核。

**为什么不能靠「跑一遍看文件变没变」来核**：`dump-config` 写回的是**同一份内容**
（磁盘配置本就是 `DEFAULTS` 的副本）⇒ 任何基于 md5 的核对都把它判成只读。
**哈希能证明「没变」，不能证明「没写」。**
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from . import emit as E
from . import expectations as EXP
from . import invariants as INV
from . import visualize as VIS
from .config import ConfigError, dump_config, load_config
from .graph import load_graphs
from .ir import Graph
from .pipeline import run_pipeline

#: 项目根：`src/tensorcompiler/cli.py` → 上两级。
_PKG_ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def _quiet_prose(enabled: bool) -> Iterator[None]:
    """`--json` 时把命令里的 `print` 吞掉（stdout 只许有一段 JSON）。

    ⚠️ 必须是 `@contextlib.contextmanager` 装饰的**生成器**，不能只写生成器 ——
    只写生成器时 `with _quiet_prose(...)` 抛的是
    `TypeError: 'generator' object does not support the context manager protocol`，
    而这条错误发生在**每一条**子命令上，看起来像 CLI 整体坏了。
    """
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _emit_json(obj: Any) -> None:
    """走**真 stdout**（`_quiet_prose` 只吞 `print`，吞不掉这个）。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=1,
                                sort_keys=True) + "\n")


def _anchor_args(args: argparse.Namespace) -> argparse.Namespace:
    """把**所有参与路径解析**的参数锚到项目根，而不是 cwd。

    判据不是「调用点写没写 `root=`」，而是「**这个参数有没有参与路径解析**」。
    ⇒ 同一 CLI 里两个开关的锚定约定必须一致（`--config` / `--out` / `--target`）。
    """
    for attr in ("config", "out", "target"):
        v = getattr(args, attr, None)
        if isinstance(v, str) and v and not Path(v).is_absolute():
            setattr(args, attr, str((_PKG_ROOT / v).resolve()))
    return args


def _fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """补回 `argparse.SUPPRESS` 吞掉的开关默认值。

    ⚠️ 子命令的 `--config/--out/--json` 都写成 `default=argparse.SUPPRESS`
    （这样「父级给的值」不会被子命令的默认值盖掉）。代价是**没给这个开关时，
    属性根本不存在** —— 于是 `args.json` 抛 `AttributeError` 而不是取到 `False`。

    这件事必须**显式**做一次：靠 `getattr(args, "json", False)` 散落在
    七个读点里，等于把「有没有默认值」这个决定复制了七遍，漏掉一处就是
    「这条子命令崩、别的都不崩」。
    """
    for name, default in (("json", False), ("config", None), ("out", None)):
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def _load_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, root=_PKG_ROOT)


def _graphs_from_products(products: dict[str, Any]) -> tuple[Graph, ...]:
    return tuple(Graph.from_dict(d) for d in products["graphs.json"]["graphs"])


# --------------------------------------------------------------------- 子命令
def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    with _quiet_prose(args.json):
        run = run_pipeline(cfg, args.out, write=True)
        s = run.products["summary.json"]
        if not args.json:
            print(f"版本 {run.products['version.json']['version']}")
            print(f"图 {s['n_graphs']} 张 × 策略 {s['n_strategies']} 条 = "
                  f"{s['n_cells']} 格（缺口 {s['matrix_gap']}）")
            print(f"降级链 {s['n_stages']} 级 / pass {s['n_passes']} 个"
                  f"（可选 {s['n_optional_passes']}）")
            print(f"语义 {s['semantics']['n_bitwise_equal']}/{s['semantics']['n_checked']} 逐位相等；"
                  f"双成本互校 {s['crosscheck']['n_agree']}/{s['crosscheck']['n_checked']}")
            print(f"门槛 {s['acceptance']['n_passed']}/{s['acceptance']['n']} 通过")
            # ⚠️ 对抗用例的计数从它自己的产物读 —— `summary.json` 里**没有**它
            #    （对抗是最外一层，要拿 summary 渲染报告）。见 `pipeline._summary`。
            adv = run.products["adversarial.json"]
            print(f"对抗用例 {adv['n_held']}/{adv['n_cases']} 成立")
            print(f"期望 {s['expectations']['n_held']} 成立 / "
                  f"{s['expectations']['n_refuted']} 被推翻 / "
                  f"{s['expectations']['n_unmeasurable']} 算不出来")
            print(f"产物目录：{run.out_dir}")
    if args.json:
        _emit_json({"version": run.products["version.json"]["version"],
                    "summary": run.products["summary.json"]})
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """只读：从 `out/` 重跑验收与对抗用例（rc=1 表示有门槛未通过）。"""
    cfg = _load_cfg(args)
    out = Path(args.out)
    if not (out / "summary.json").exists():
        print(f"❌ {out} 里没有产物 —— 先跑 `python main.py run`")
        return 1
    products = E.read_products(out)
    products.setdefault("version.json", {"version": "?",
                                         "versioned_files": list(E.VERSIONED_FILES)})
    graphs = _graphs_from_products(products)
    acc = INV.run_acceptance(products, cfg)
    adv = INV.run_adversarial(products, cfg, graphs)
    exp = EXP.evaluate(products, cfg)
    with _quiet_prose(args.json):
        if not args.json:
            for r in acc["rows"]:
                mark = "OK  " if r["passed"] else "FAIL"
                print(f"  [{mark}] {r['key']:42s} {r['label']}；实测 {r['observed']:.6g}")
            print(f"对抗用例 {adv['n_held']}/{adv['n_cases']} 成立（**不可调**的不变量断言）")
            for r in adv["cases"]:
                if not r["held"]:
                    print(f"  [违反] {r['name']}: {r['detail']}")
            print(f"期望：{exp['n_held']} 成立 / {exp['n_refuted']} 被推翻 / "
                  f"{exp['n_unmeasurable']} 算不出来")
            for r in exp["rows"]:
                if r["verdict"] != "held":
                    print(f"  [{r['verdict']:12s}] {r['id']}: {json.dumps(r['observed'], ensure_ascii=False)[:200]}")
            print("只读：未写盘。")
    if args.json:
        _emit_json({"acceptance": acc, "adversarial": adv, "expectations": exp})
    ok = acc["n_failed"] == 0 and adv["n_violated"] == 0
    return 0 if ok else 1


def cmd_visualize(args: argparse.Namespace) -> int:
    """只读：从 `out/` 重建报告并逐位比对（`--target` 时才另写他处）。"""
    out = Path(args.out)
    if not (out / "summary.json").exists():
        print(f"❌ {out} 里没有产物 —— 先跑 `python main.py run`")
        return 1
    products, md, html = VIS.rebuild(out)
    res = VIS.verify(out)
    if getattr(args, "target", None):
        t = Path(args.target)
        t.mkdir(parents=True, exist_ok=True)
        (t / "report.md").write_text(md, encoding="utf-8", newline="\n")
        (t / "report.html").write_text(html, encoding="utf-8", newline="\n")
    with _quiet_prose(args.json):
        if not args.json:
            for r in res["rows"]:
                mark = "逐位相同" if r["bitwise_equal"] else "**不同**"
                print(f"  {r['file']:14s} {mark:10s} {r['bytes']} 字节")
            print("结论：" + ("**产物足以重造报告**" if res["all_equal"]
                            else "**重建的报告与盘上的不同**"))
    if args.json:
        _emit_json(res)
    return 0 if res["all_equal"] else 1


def cmd_verify(args: argparse.Namespace) -> int:
    """只读：确定性与产物充分性核对（只写临时目录，跑完删）。"""
    cfg = _load_cfg(args)
    out = Path(args.out)
    with tempfile.TemporaryDirectory(prefix="tcf-verify-") as td:
        a = run_pipeline(cfg, Path(td) / "a", write=True).products
        b = run_pipeline(cfg, Path(td) / "b", write=True).products
    expected_diff = set(E.EXCLUDED) | {"version.json"}
    rows = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            rows.append({"file": name, "same": False, "note": "只在一侧出现"})
            continue
        same = E.dumps(a[name]) == E.dumps(b[name])
        # ⚠️ `bytes` 是**真的字节数**：`E.dumps` 返回 `str`，`len()` 是字符数。
        #    报告与产物里全是中文，两者差 1.3 倍以上（C.345）。
        blob = E.dumps(a[name]).encode("utf-8")
        rows.append({"file": name, "same": same, "bytes": len(blob),
                     "chars": len(blob.decode("utf-8")),
                     "expected_to_differ": name in expected_diff})
    unexpected = [r for r in rows if not r["same"] and not r["expected_to_differ"]]
    # 产物充分性：从盘上产物重建报告 == 直接渲染
    suff = VIS.verify(out) if (out / "summary.json").exists() else {"all_equal": None, "rows": []}
    with _quiet_prose(args.json):
        if not args.json:
            print(f"版本 {a['version.json']['version']} / "
                  f"{b['version.json']['version']}"
                  f"（相同={a['version.json']['version'] == b['version.json']['version']}）")
            for r in rows:
                if r.get("note"):
                    print(f"  {r['file']:20s} {r['note']}")
                    continue
                mark = "逐位相同" if r["same"] else "**不同**"
                print(f"  {r['file']:20s} {mark:10s} {r.get('bytes', 0)} 字节")
            print(f"预期不同的文件：{sorted(n for n in expected_diff if n in a)}")
            print(f"**意外不同**的文件：{[r['file'] for r in unexpected] or '（无）'}")
            print(f"产物充分性（从产物重建报告 == 直接渲染）：{suff['all_equal']}")
            print("结论：" + ("**全部通过**" if not unexpected and suff["all_equal"] is not False
                            else "**有核对未通过**"))
    if args.json:
        _emit_json({"rows": rows, "unexpected": [r["file"] for r in unexpected],
                    "sufficiency": suff["all_equal"],
                    "expected_to_differ": sorted(expected_diff)})
    return 0 if not unexpected and suff["all_equal"] is not False else 1


def cmd_show(args: argparse.Namespace) -> int:
    """只读：打印摘要。"""
    out = Path(args.out)
    p = out / "summary.json"
    if not p.exists():
        print(f"❌ {out} 里没有产物 —— 先跑 `python main.py run`")
        return 1
    products = E.read_products(out)
    s = products["summary.json"]
    with _quiet_prose(args.json):
        if not args.json:
            print(f"版本 {products['version.json']['version']}")
            print(f"图 {s['n_graphs']} 张 / 策略 {s['n_strategies']} 条 / "
                  f"{s['n_cells']} 格（缺口 {s['matrix_gap']}）")
            print(f"门槛 {s['acceptance']['n_passed']}/{s['acceptance']['n']}；"
                  f"对抗用例 {s['adversarial']['n_held']}/{s['adversarial']['n_cases']}；"
                  f"期望 {s['expectations']['n_held']} 成立 / "
                  f"{s['expectations']['n_refuted']} 推翻")
            for pg in products["compare.json"]["per_graph"]:
                b = pg["baseline"]["readings"]
                bl = pg["best_latency"]["readings"]
                print(f"  {pg['graph']:12s} 基线延迟 {b['latency_us']:.3f} µs → "
                      f"最优 {bl['latency_us']:.3f} µs（{pg['best_latency']['describe']}）")
    if args.json:
        _emit_json(products["summary.json"])
    return 0


def cmd_dump_config(args: argparse.Namespace) -> int:
    """⚠️ 会写盘：把当前生效的配置写回 `configs/default.json`。"""
    cfg = _load_cfg(args)
    target = Path(args.config)
    dump_config(cfg, target)
    with _quiet_prose(args.json):
        if not args.json:
            print(f"已把当前生效的配置写回：{target}")
            print("（`deep_merge` 里**列表整体替换、磁盘优先** ⇒ "
                  "改完 `config.py` 的 `DEFAULTS` 后必须跑这一条，"
                  "否则磁盘上的旧列表会静默盖掉新声明。）")
    if args.json:
        _emit_json({"written": str(target)})
    return 0


# --------------------------------------------------------------------- 解析
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tensorcompiler",
        description="TensorCompilerFactory：Tensor Graph → IR → Fusion → Kernel 的降级链",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", default=argparse.SUPPRESS, help="配置 JSON")
        sp.add_argument("--out", default=argparse.SUPPRESS, help="产物目录")
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="stdout 上只输出一段 JSON")

    # ⚠️ `help=` 只出现在**父级列表**里；子命令自己的 `--help` 只印 usage + options。
    #    `description=` 补上后，`main.py <cmd> --help` 才会印出那句话 ——
    #    于是「这个命令写不写盘」这件事**在两条路径上都能读到**。
    def add(name: str, desc: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=desc, description=desc)
        common(sp)
        return sp

    # ⚠️ 每个 `desc` 里必须**恰好出现一个**显式标记：`会写盘` 或 `只读`。
    add("run", "跑完整流水线（会写盘：写 out/）")

    vis = add("visualize", "从 out/ 重建报告并逐位比对（只读；--target 时才另写他处）")
    vis.add_argument("--target", default=argparse.SUPPRESS,
                     help="需要落盘时写到这个目录（默认不写盘）")

    add("check", "从 out/ 重跑验收与对抗用例（只读）")
    add("verify", "确定性与产物充分性核对（只读：只写临时目录）")
    add("show", "打印摘要（只读）")
    add("dump-config", "⚠️ 会写盘：把当前生效的配置写回 configs/default.json")

    p.add_argument("--config", default="configs/default.json", help=argparse.SUPPRESS)
    p.add_argument("--out", default="out", help=argparse.SUPPRESS)
    return p


_DISPATCH = {
    "run": cmd_run,
    "check": cmd_check,
    "visualize": cmd_visualize,
    "verify": cmd_verify,
    "show": cmd_show,
    "dump-config": cmd_dump_config,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # ⚠️ 锚定**一次**（紧跟 parse_args），不在每个 load_config 调用点各锚一次 ——
    #    后者会让「哪个参数被锚了」变成「调用点写没写 root=」，而漏掉一个
    #    的症状是「从工作区根调用时静默换了一份配置」。
    args = _anchor_args(args)
    args = _fill_defaults(args)
    try:
        return _DISPATCH[args.cmd](args)
    except ConfigError as e:
        print(f"❌ 配置错误：{e}", file=sys.stderr)
        return 2
