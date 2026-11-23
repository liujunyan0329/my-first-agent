"""阶段 5 · 把 trace / eval / caching / routing 的结果汇总成一份成本报告。

输入（都在 stage5/runtime/ 下）：
  eval_baseline.json   30 题基线的通过率、token、费用、失败模式
  index_eval.json      切块 / top-k / 检索方式的消融结果
  caching.json         prompt caching 的命中率与节省
  routing.json         模型路由的三种策略对比

输出：
  stage5/runtime/cost_report.md   可以直接贴进项目 README 的报告

运行：
  uv run python stage5/cost_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"


def load(name: str) -> dict[str, Any] | None:
    path = RUNTIME / name
    if not path.exists():
        print(f"[跳过] {path} 不存在，先运行对应实验")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value:.0%}"


def money(value: float) -> str:
    return f"¥{value:.4f}"


def build_report() -> str:
    # 优先用消融实验里的基线，保证和第 2 节是同一批运行；没有再退回 rag_eval 的输出。
    baseline = load("eval_vector500_k4.json") or load("eval_baseline.json")
    index_eval = load("index_eval.json")
    caching = load("caching.json")
    routing = load("routing.json")

    lines: list[str] = []
    lines.append("# 阶段 5 · 质量与成本报告")
    lines.append("")
    lines.append("> 本文件由 `stage5/cost_report.py` 根据本机实测数据生成。")
    lines.append("> 模型：deepseek-chat / deepseek-reasoner；时间为 2026-11。")
    lines.append("")

    if baseline:
        lines.append("## 1. 基线质量（30 条 eval）")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---|")
        lines.append(f"| 关键词通过 | {baseline['keyword_pass']}/{baseline['questions']} "
                     f"({pct(baseline['keyword_pass_rate'])}) |")
        lines.append(f"| judge 通过 | {baseline['judge_pass']}/{baseline['questions']} "
                     f"({pct(baseline['judge_pass_rate'])}) |")
        lines.append(f"| 双信号通过 | {baseline['final_pass']}/{baseline['questions']} "
                     f"({pct(baseline['final_pass_rate'])}) |")
        lines.append(f"| 平均 judge | {baseline['avg_judge']:.2f} / 5 |")
        lines.append(f"| 平均 context 覆盖 | {baseline['avg_context_coverage']:.2f} |")
        lines.append(f"| token / 费用 / 耗时 | "
                     f"{baseline['prompt_tokens'] + baseline['completion_tokens']} tok / "
                     f"{money(baseline['cost'])} / {baseline['seconds']:.1f}s |")
        lines.append("")
        lines.append(
            "> 本节来自 `eval_vector500_k4.json` 的一次独立重跑；第 2 节的消融基线"
            "是另一批运行。同一配置多次运行会在 20~22/30 之间波动，"
            "这是 eval 噪音，不是数据错误。"
        )
        lines.append("")
        modes = baseline.get("failure_modes", {})
        lines.append(f"**失败模式**：`{modes}`")
        failed = [r for r in baseline.get("records", []) if not r["final_pass"]]
        if failed:
            lines.append("")
            lines.append("| 题号 | 模式 | 关键词 | judge | 摘录 |")
            lines.append("|---|---|---|---|---|")
            for record in failed:
                answer = str(record["answer"]).replace("|", "\\|")[:60]
                lines.append(
                    f"| {record['qid']} | {record['failure_mode']} | "
                    f"{record['keyword_coverage']:.2f} | {record['judge_score']} | "
                    f"{answer}… |"
                )
        lines.append("")

    if index_eval:
        lines.append("## 2. 切块 / top-k / 检索方式消融")
        lines.append("")
        lines.append("| 配置 | 块数 | top-k | 方式 | 关键词 | judge | 双通过 | Δ | 费用 | 耗时 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        base = index_eval["results"][0]
        for result in index_eval["results"]:
            delta = (result["final_pass_rate"] - base["final_pass_rate"]) * 100
            lines.append(
                f"| {result['label']} | {result['chunks']} | {result['top_k']} | "
                f"{result['mode']} | {pct(result['keyword_pass_rate'])} | "
                f"{pct(result['judge_pass_rate'])} | {pct(result['final_pass_rate'])} | "
                f"{delta:+.1f}pp | {money(result['cost'])} | {result['seconds']:.1f}s |"
            )
        lines.append("")
        lines.append(
            "注意：30 题的样本很小，差 1~2 题就是 3~7 个百分点；"
            "这里的排序用于排除明显无效的改动，不是统计显著性结论。"
        )
        lines.append("")

    if caching:
        rows = caching["fixed_prefix"]
        # 第一次调用必然 miss（缓存还没写入），所以"第 2 次起"的命中率要排除第一行。
        hit_ratios = [
            row["hit"] / row["prompt"] for row in rows[1:] if row["prompt"]
        ]
        total = sum(row["cost"] for row in rows)
        uncached = sum(row["uncached_cost"] for row in rows)
        lines.append("## 3. prompt caching")
        lines.append("")
        lines.append(f"- 固定前缀长度：{caching['long_prefix_chars']} 字")
        lines.append(f"- 第 2 次起的平均缓存命中率：{sum(hit_ratios) / len(hit_ratios):.0%}")
        lines.append(f"- 总费用：{money(total)}；不缓存：{money(uncached)}；"
                     f"节省 {1 - total / uncached:.0%}")
        changed = caching["changed_prefix"]
        lines.append(f"- 每次改前缀开头标记：命中率 "
                     f"{sum(r['hit'] / r['prompt'] for r in changed) / len(changed):.0%}"
                     f"（验证前缀一变缓存全废）")
        lines.append("")
        lines.append("| 第几次 | prompt | 命中 | 未命中 | 费用 | 不缓存费用 | 耗时 |")
        lines.append("|---|---|---|---|---|---|---|")
        for index, row in enumerate(rows, start=1):
            lines.append(
                f"| {index} | {row['prompt']} | {row['hit']} | {row['miss']} | "
                f"{money(row['cost'])} | {money(row['uncached_cost'])} | "
                f"{row['seconds']:.2f}s |"
            )
        lines.append("")

    if routing:
        lines.append("## 4. 模型路由")
        lines.append("")
        lines.append("| 策略 | 双通过 | 关键词 | judge | 平均 judge | 生成费用 | 总费用 | 耗时 | 模型构成 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for result in routing["results"]:
            lines.append(
                f"| {result['strategy']} | {result['final_pass']}/{result['questions']} | "
                f"{pct(result['keyword_pass'] / result['questions'])} | "
                f"{pct(result['judge_pass'] / result['questions'])} | "
                f"{result['avg_judge']:.2f} | {money(result['gen_cost'])} | "
                f"{money(result['total_cost'])} | {result['seconds']:.1f}s | "
                f"{result['calls_by_model']} |"
            )
        lines.append("")
        routed = next((r for r in routing["results"] if r["strategy"] == "routed"), None)
        premium = next(
            (r for r in routing["results"] if r["strategy"] == "always_premium"), None
        )
        if routed and premium and premium["gen_cost"] > 0:
            lines.append(
                f"相对 always_premium，路由把生成成本降低了 "
                f"{1 - routed['gen_cost'] / premium['gen_cost']:.0%}；"
                f"质量 {routed['final_pass']}/{routed['questions']} vs "
                f"{premium['final_pass']}/{premium['questions']}。"
            )
            lines.append("")
            lines.append(f"路由到 premium 的题目：`{routed['premium_ids']}`")
            lines.append("")

    if routing:
        chat = next(
            (r for r in routing["results"] if r["strategy"] == "always_chat"), None
        )
        routed = next((r for r in routing["results"] if r["strategy"] == "routed"), None)
        premium = next(
            (r for r in routing["results"] if r["strategy"] == "always_premium"), None
        )
        if chat and routed and premium:
            lines.append("## 5. 每 1000 次请求的成本估算")
            lines.append("")
            lines.append("只算生成调用，不含判分（判分是离线 eval 的成本）。")
            lines.append("")
            lines.append("| 策略 | 单次生成成本 | 1000 次 |")
            lines.append("|---|---|---|")
            for result in (chat, routed, premium):
                per_call = result["gen_cost"] / result["questions"]
                lines.append(
                    f"| {result['strategy']} | {money(per_call)} | "
                    f"{money(per_call * 1000)} |"
                )
            lines.append("")

    lines.append("## 6. 已知限制")
    lines.append("")
    lines.append("- eval 只有 30 条，覆盖 5 个实验，不代表所有课程问题；")
    lines.append("- judge 与被测模型同源，存在自偏好偏差；关键词规则会漏掉同义表达；")
    lines.append("- 30 条题的检索结果与模型调用都有随机性，重跑会波动；")
    lines.append("- 价格常量按本账户 2026-11 价目表设置，换模型/账户需改常量；")
    lines.append("- prompt caching 由服务端管理，缓存可能过期，命中率不是常数；")
    lines.append("- 路由规则是手工定的，没有用独立验证集调参，存在过拟合风险。")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    report = build_report()
    path = RUNTIME / "cost_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[保存] {path}")


if __name__ == "__main__":
    main()
