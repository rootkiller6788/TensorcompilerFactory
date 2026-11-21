"""渲染器：**只吃产物字典**。

⚠️ 这条约束本身就是一条不变量（`renderers_read_nothing_but_the_products`）：
`render(products)` 只有一个入参，一行不读世界 / 模型 / 真值。

买到两样：
1. `visualize` 从 `out/` 重建报告 —— 不需要重跑流水线；
2. 命题从「流水线确定」升级为「**产物足以重造报告**」。
"""

from __future__ import annotations

from typing import Any

from . import cost as C

# --------------------------------------------------------------------- 工具
def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _f(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def _g(x: float) -> str:
    return f"{x:.6g}"


def _ratio(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}×"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return out


# --------------------------------------------------------------------- 正文
def _head(p: dict[str, Any]) -> list[str]:
    s = p["summary.json"]
    v = p["version.json"]["version"]
    L: list[str] = []
    L.append("# TensorCompilerFactory · 编译报告")
    L.append("")
    L.append(f"> 版本 `{v}` ｜ 协议指纹 `{s['protocol_digest']}` ｜ "
             f"{s['n_graphs']} 张图 × {s['n_strategies']} 条降级策略 = "
             f"**{s['n_cells']} 格** ｜ 矩阵缺口 **{s['matrix_gap']}**")
    L.append("")
    L.append("## 一句话结论")
    L.append("")
    L.append(_headline(p))
    L.append("")
    return L


def _headline(p: dict[str, Any]) -> str:
    cmp_ = p["compare.json"]
    att = p["attribution.json"]
    exp = p["expectations.json"]
    rows = []
    for pg in cmp_["per_graph"]:
        b = pg["baseline"]["readings"]
        best = pg["best_latency"]["readings"]
        rows.append((pg["graph"], b, best, pg["best_latency"]["describe"]))
    parts = []
    for name, b, best, desc in rows:
        r = b["latency_us"] / best["latency_us"] if best["latency_us"] else 0.0
        parts.append(f"`{name}` **{r:.2f}×**（{desc}）")
    return ("同一批张量图，最差的降级路径与最好的之间差："
            + "；".join(parts)
            + f"。全链共 **{att['n_passes']}** 个 pass，其中 "
              f"**{att['n_load_bearing_some']}** 个在至少一张图上真的有杀伤；"
              f"设计期望 **{exp['n_refuted']}/{exp['n']}** 条被实测推翻。")


def _section_proposition(p: dict[str, Any]) -> list[str]:
    L = ["## 核心命题", ""]
    L.append("**这个项目不是「写一个编译器」。** 真正研究的是：")
    L.append("")
    L.append("> 一个 Tensor Program 是怎么**一步步**降级成 Hardware Kernel 的 —— "
             "每一级**做了什么、省了多少**。")
    L.append("")
    L.append("所以主证据不是「我实现了一个编译器」，而是 **漏斗差额**：")
    L.append("沿 pass 链，程序还要做多少活。")
    L.append("")
    return L


def _section_shape(p: dict[str, Any]) -> list[str]:
    L = ["## 形状判定：**直线流水线的第二形态（编译器降级链）**", ""]
    L.append("| | 第一形态（DatasetFactory） | **第二形态（本项目）** |")
    L.append("| --- | --- | --- |")
    L.append("| 对象怎么变 | **被筛**（过滤 / 切分） | **被改写**（重写 / 降级） |")
    L.append("| 漏斗差额问的是 | 多少数据活下来 | **程序还要做多少活** |")
    L.append("| 真值 | 数据本身 | **语义**（真跑一遍逐位相等） |")
    L.append("")
    L.append("**它不是**扇出-收拢型（候选是配置里交叉积展开的，不是搜索器挑的；"
             "产物是漏斗与归因，不是前沿），**也不是**比较型"
             "（比较型什么都不产出，而编译器**产出一个程序**）。")
    L.append("")
    L.append("它带来三样家族里没有的东西：")
    L.append("")
    L.append("1. **每一级 IR 都是产物** —— 变换链本身可审计，不只是首尾两点；")
    L.append("2. **语义保持是可执行真值** —— 降级前后真跑一遍，逐位相等；")
    L.append("3. **双成本互校 + 归因** —— 静态解析式 vs 动态访问计数；"
             "消融一个 pass 看漏斗怎么变。")
    L.append("")
    return L


def _section_funnel(p: dict[str, Any]) -> list[str]:
    L = ["## 漏斗：逐级降级链", ""]
    L.append("下表每一行是**一级 pass**，每一列是一个读数。"
             "`as_written` 是朴素降级（每个算子一个核、不分块）。")
    L.append("")
    # ⚠️ `sorted(...)`：产物经 JSON 往返后键序变成**字典序**，
    #    而内存里的 dict 是**插入序**。不排序 ⇒ 「渲染两次相同」仍然绿
    #    （两次都用同一份内存对象），但「从盘上重建 == 直接渲染」不成立。
    #    ⇒ 这正是 `visualize` 那条只读命令存在的理由：**它比纯函数性更强**。
    for gname, rows in sorted(p["ladder.json"]["per_graph"].items()):
        L.append(f"### `{gname}`")
        L.append("")
        L.append(_stage_table(rows))
        L.append("")
    # ---- 基线链：什么都不开 --------------------------------------------
    off = p["ladder.json"].get("baseline")
    if off:
        L.append("### 基线链：**什么都不开**")
        L.append("")
        L.append("参考链是全开的 ⇒ 里面**一条 `skipped` 级都没有**，"
                 "而「跳过就是没做」这条断言必须**真的被执行到**才有意义。"
                 "下面这条链把布尔开关全部关掉、枚举轴取各自最省的一档："
                 "它同时回答「编译器到底做了多少事」—— 这条链的读数就是"
                 "「什么都不做」的那一端。")
        L.append("")
        L.append("⚠️ `skipped` 说的是「这个 pass 有没有被**跑**」，不是"
                 "「它有没有**改变**东西」：`tiling=none` 时 tiling pass 跑了、"
                 "决定不分块 ⇒ `skipped=False`。")
        L.append("")
        L.append(f"策略：`{_dump(p['ladder.json'].get('baseline_strategy', {}))}`")
        L.append("")
        for gname, rows in sorted(off.items()):
            L.append(f"#### `{gname}`（基线链）")
            L.append("")
            L.append(_stage_table(rows))
            L.append("")
    return L


def _stage_table(rows: list[dict]) -> str:
    hdr = ["级", "核数", "全局访存元素", "shared 字节", "寄存器",
           "占用率", "延迟 µs", "这一级做了什么"]
    body = []
    for r in rows:
        rd = r["readings"]
        note = _stage_note(r)
        body.append([f"`{r['name']}`", str(r["n_kernels"]),
                     f"{rd['global_elems']:,}", f"{rd['peak_shared_bytes']:,}",
                     str(rd["peak_registers"]), _f(rd["occupancy"], 3),
                     _f(rd["latency_us"], 3), note])
    return "\n".join(_table(hdr, body))


def _stage_note(r: dict) -> str:
    m = r["meta"]
    if r.get("skipped"):
        return "**跳过**（策略里关了）"
    if "n_saved_ops" in m:
        return f"删掉 {m['n_saved_ops']} 个恒等算子"
    if "n_shapes_checked" in m:
        return f"互校 {m['n_shapes_checked']} 个声明形状"
    if "n_folded" in m:
        return f"折叠 {m['n_folded']} 个常量节点"
    if "n_merged" in m:
        return f"合并 {m['n_merged']} 个重复子表达式"
    if "n_kernels_after" in m:
        return f"核 {m['n_kernels_before']} → {m['n_kernels_after']}"
    if "tile" in m:
        return f"tile = {m['tile']}"
    if "layout" in m:
        return (f"布局 {m['layout']}（shared ×{m['shared_scale']}、"
                f"寄存器 +{m['extra_registers']}、shared 带宽 ×{m['shared_bandwidth_scale']}）")
    if "memory_plan" in m:
        return f"内存规划 {m['memory_plan']}"
    if "threads" in m:
        return f"线程映射 {m['threads']} 线程/块"
    return "—"


def _section_compare(p: dict[str, Any]) -> list[str]:
    L = ["## 多策略对比", ""]
    cm = p["compare.json"]
    L.append(f"{cm['n_graphs']} 张图 × {cm['n_strategies']} 条策略 = "
             f"**{cm['matrix_expected_total']} 格**，缺口 **{cm['matrix_gap_total']}**。")
    L.append("")
    hdr = ["图", "算子数", "基线延迟 µs", "最优延迟 µs", "降幅",
           "基线访存", "最优访存", "降幅", "前沿点数", "最优策略"]
    body = []
    for pg in cm["per_graph"]:
        b = pg["baseline"]["readings"]
        bl = pg["best_latency"]["readings"]
        bb = pg["best_bytes"]["readings"]
        body.append([f"`{pg['graph']}`", str(pg["n_ops"]),
                     _f(b["latency_us"], 3), _f(bl["latency_us"], 3),
                     _ratio(b["latency_us"] / bl["latency_us"] if bl["latency_us"] else None),
                     f"{b['global_bytes']:,}", f"{bb['global_bytes']:,}",
                     _ratio(b["global_bytes"] / bb["global_bytes"] if bb["global_bytes"] else None),
                     str(pg["n_pareto"]), f"`{pg['best_latency']['describe']}`"])
    L.extend(_table(hdr, body))
    L.append("")
    L.append("### 前沿")
    L.append("")
    for pg in cm["per_graph"]:
        L.append(f"**`{pg['graph']}`**（{pg['n_pareto']} 个权衡点）")
        L.append("")
        # ⚠️ `extend` 不是 `append`：`_table` 返回的是**行列表**，
        #    `append` 会把整个列表当成一行塞进 md_lines，
        #    症状是 `TypeError: sequence item 73: expected str instance, list found`
        #    —— 报错发生在 `"\n".join` 那一刻，离真正写错的地方隔了 40 行。
        L.extend(_table(["策略", "访存字节", "延迟 µs", "峰值内存字节"],
                        [[f"`{x['describe']}`",
                          f"{x['readings']['global_bytes']:,}",
                          _f(x["readings"]["latency_us"], 3),
                          f"{x['readings']['peak_bytes']:,}"]
                         for x in pg["pareto"]]))
        L.append("")
    return L


def _section_attribution(p: dict[str, Any]) -> list[str]:
    L = ["## 归因：消融一个 pass 看漏斗怎么变", ""]
    at = p["attribution.json"]
    L.append("「能解释每一次 lowering 选择带来的性能差异」不是写一段散文 —— "
             "是**把那个 pass 关掉，看读数变不变**。")
    L.append("")
    for r in at["per_graph"]:
        L.append(f"### `{r['graph']}`")
        L.append("")
        L.extend(_table(["pass", "关掉后的核数", "有变化的读数", "有杀伤"],
                        [[f"`{name}`", str(d["n_kernels_without_it"]),
                          ", ".join(f"`{x}`" for x in d["readings_changed"]) or "**（无）**",
                          "是" if d["load_bearing"] else "**否（装饰）**"]
                         for name, d in sorted(r["per_pass"].items())]))
        L.append("")
        L.append(f"- 承重的 pass：{', '.join('`'+x+'`' for x in r['load_bearing_passes']) or '**（无）**'}")
        L.append(f"- 装饰的 pass：{', '.join('`'+x+'`' for x in r['decorative_passes']) or '（无）'}")
        L.append(f"- 逐项可加的读数：**{r['n_additive_readings']}/{r['n_readings']}**"
                 + (f"；不可加的：{', '.join('`'+k+'`' for k in sorted(r['is_additive']) if not r['is_additive'][k])}"
                    if r["n_additive_readings"] < r["n_readings"] else ""))
        L.append("")
    return L


def _section_checks(p: dict[str, Any]) -> list[str]:
    L = ["## 三层核对", ""]
    s = p["summary.json"]
    L.append(f"- **语义**：{s['semantics']['n_bitwise_equal']}/"
             f"{s['semantics']['n_checked']} 逐位相等，最大漂移 "
             f"**{s['semantics']['max_abs_drift']:.3e}**")
    L.append(f"- **双成本互校**：{s['crosscheck']['n_agree']}/"
             f"{s['crosscheck']['n_checked']} 静态 == 动态")
    L.append(f"- **门槛**：{s['acceptance']['n_passed']}/{s['acceptance']['n']} 通过")
    # ⚠️ 对抗用例的计数**从它自己的产物里读**，不从 summary 读 ——
    #    summary 在对抗之前就要算好（对抗要拿它渲染报告），
    #    所以它不可能包含对抗的结果。见 `pipeline._summary` 的注释。
    adv = p["adversarial.json"]
    L.append(f"- **对抗用例**：{adv['n_held']}/{adv['n_cases']} 成立")
    L.append(f"- **设计期望**：{s['expectations']['n_held']} 成立 / "
             f"{s['expectations']['n_refuted']} 被推翻 / "
             f"{s['expectations']['n_unmeasurable']} 算不出来")
    L.append("")
    L.append("### 门槛")
    L.append("")
    L.extend(_table(["门槛", "要求", "实测", "结论"],
                    [[f"`{r['key']}`", r["label"],
                      _g(r["observed"]), "通过" if r["passed"] else "**未通过**"]
                     for r in p["acceptance.json"]["rows"]]))
    L.append("")
    L.append("### 对抗用例（**不可调**的不变量断言）")
    L.append("")
    L.extend(_table(["用例", "断言", "结论", "读数"],
                    [[f"`{r['name']}`", r["claim"],
                      "成立" if r["held"] else "**违反**", r["detail"]]
                     for r in p["adversarial.json"]["cases"]]))
    L.append("")
    L.append("### 设计期望")
    L.append("")
    L.extend(_table(["期望", "声明", "裁定"],
                    [[f"`{r['id']}`", r["claim"],
                      {"held": "成立", "refuted": "**被推翻**",
                       "unmeasurable": "算不出来"}[r["verdict"]]]
                     for r in p["expectations.json"]["rows"]]))
    L.append("")
    return L


def _section_footer(p: dict[str, Any]) -> list[str]:
    L = ["## 产物清单", ""]
    L.extend(_table(["文件", "大小（字节）", "进版本摘要"],
                    [[f"`{k}`", str(len(_dump(v))), "是" if k in
                      p["version.json"]["versioned_files"] else "否"]
                     for k, v in sorted(p.items())]))
    L.append("")
    L.append(f"版本 `{p['version.json']['version']}`，"
             f"创建于 `{p['version.json']['created_at']}`。")
    L.append("")
    return L


def _dump(v: Any) -> str:
    import json
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------- 入口
def render(products: dict[str, Any]) -> tuple[str, str]:
    """`products` → `(markdown, html)`。**纯函数**。"""
    md_lines: list[str] = []
    md_lines += _head(products)
    md_lines += _section_proposition(products)
    md_lines += _section_shape(products)
    md_lines += _section_funnel(products)
    md_lines += _section_compare(products)
    md_lines += _section_attribution(products)
    md_lines += _section_checks(products)
    md_lines += _section_footer(products)
    md = "\n".join(md_lines) + "\n"
    return md, _to_html(md, products)


# --------------------------------------------------------------------- HTML
_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
       max-width: 1080px; margin: 2rem auto; padding: 0 1.2rem; line-height: 1.65; }
h1 { font-size: 1.7rem; border-bottom: 2px solid currentColor; padding-bottom: .3rem; }
h2 { font-size: 1.3rem; margin-top: 2.2rem; border-bottom: 1px solid #8884; padding-bottom: .2rem; }
h3 { font-size: 1.08rem; margin-top: 1.6rem; }
table { border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .88rem; }
th, td { border: 1px solid #8884; padding: .32rem .55rem; text-align: left;
         vertical-align: top; }
th { background: #8881; }
code { background: #8881; padding: .08rem .3rem; border-radius: 3px;
       font-family: ui-monospace, Consolas, monospace; font-size: .9em; }
blockquote { border-left: 3px solid #8886; margin: .8rem 0; padding: .2rem 1rem;
             background: #8881; }
strong { font-weight: 600; }
"""


def _to_html(md: str, products: dict[str, Any]) -> str:
    """极简 Markdown → HTML（**只支持本项目用到的语法**）。

    ⚠️ 不引第三方 Markdown 库的理由与核心路径一致（零依赖）；
    但因此它**只支持自己生成的语法** —— 拿它渲染别人的 Markdown 会静默丢格式。
    这条限制写在注释里，不是「大概能用」。
    """
    import html as _h
    out: list[str] = []
    in_table = False
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("| "):
            cells = [c.strip() for c in s.strip("|").split(" | ")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>"
                                        for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if s.startswith("### "):
            out.append(f"<h3>{_inline(s[4:])}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{_inline(s[3:])}</h2>")
        elif s.startswith("# "):
            out.append(f"<h1>{_inline(s[2:])}</h1>")
        elif s.startswith("> "):
            out.append(f"<blockquote>{_inline(s[2:])}</blockquote>")
        elif s.startswith("- "):
            out.append(f"<div>&bull; {_inline(s[2:])}</div>")
        elif s.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{_inline(s)}</p>")
    if in_table:
        out.append("</table>")
    body = "\n".join(out)
    title = f"TensorCompilerFactory · {products['version.json']['version']}"
    return (f"<!DOCTYPE html>\n<html lang=\"zh\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_h.escape(title)}</title><style>{_CSS}</style></head>"
            f"<body>\n{body}\n</body></html>\n")


def _inline(s: str) -> str:
    import html as _h
    import re
    s = _h.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s
