"""阶段 5 · 模型路由：简单问题走便宜模型，难题走贵模型。

策略对比（同一套 30 题、同一份检索结果、同一个判分器）：

  1. always_chat     ：全部 deepseek-chat（便宜、快）
  2. always_premium  ：全部 deepseek-reasoner（贵、慢、带思维链）
  3. routed          ：规则分类器决定用哪个

路由器看得只有一个很便宜的信号：
  - 题干里有没有"为什么 / 区别 / 相比 / 是否意味着"这类词；
  - 题干是否特别长（> 55 字）。

这是一个**故意做得很简单**的路由器。真实系统可能用一个小模型做分类，
或者用检索置信度。重点不是路由器多聪明，而是：

  路由的收益 = 难题上的质量提升 × 难题占比 − 便宜模型本来就够用的部分。

如果便宜模型在所有题上都一样好，路由只会增加复杂度，不会省钱。

运行：
  uv run python stage5/routing_demo.py
  uv run python stage5/routing_demo.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from rag_eval import (  # noqa: E402
    Question,
    Usage,
    coverage,
    generate,
    judge,
    load_questions,
    make_client,
)
from retrieval_v2 import Retriever, format_context  # noqa: E402

RUNTIME = ROOT / "runtime"

CHAT = "deepseek-chat"
PREMIUM = "deepseek-reasoner"
ROUTER_TOP_K = 4

# 价格表（¥ / token）。换模型或换账户价目表时只改这里。
PRICES: dict[str, tuple[float, float, float]] = {
    # 输入未命中, 输入命中, 输出
    CHAT: (2.0 / 1_000_000, 0.2 / 1_000_000, 8.0 / 1_000_000),
    PREMIUM: (4.0 / 1_000_000, 1.0 / 1_000_000, 16.0 / 1_000_000),
}

HARD_MARKERS = ("为什么", "区别", "相比", "比较", "推导", "是否意味着", "论证")


def cost_for(model: str, prompt: int, completion: int, cached: int) -> float:
    miss_price, hit_price, output_price = PRICES[model]
    hit = min(max(cached, 0), max(prompt, 0))
    return (prompt - hit) * miss_price + hit * hit_price + completion * output_price


def route(question: Question) -> tuple[str, str]:
    """返回 (模型, 理由)。这是本实验唯一的路由逻辑。"""
    markers = [word for word in HARD_MARKERS if word in question.question]
    if markers:
        return PREMIUM, f"命中推理标记 {markers}"
    if len(question.question) > 55:
        return PREMIUM, f"题干较长（{len(question.question)} 字）"
    return CHAT, "短事实题"


def run_strategy(
    client: Any,
    questions: list[Question],
    contexts: dict[str, str],
    *,
    name: str,
    fixed_model: str | None,
) -> dict[str, Any]:
    records = []
    usage = Usage()
    calls_by_model: dict[str, int] = {}
    started = time.perf_counter()

    print(f"\n=== 策略：{name} ===")
    print(f"{'id':<5}{'模型':<20}{'kw':>5}{'judge':>7}{'结果':>6}{'tok':>8}{'秒':>7}")
    for question in questions:
        if fixed_model:
            model, reason = fixed_model, "固定策略"
        else:
            model, reason = route(question)
        calls_by_model[model] = calls_by_model.get(model, 0) + 1

        before = Usage(usage.prompt_tokens, usage.completion_tokens,
                       usage.cached_tokens, usage.calls)
        question_started = time.perf_counter()
        answer = generate(
            client,
            question,
            contexts[question.id],
            usage,
            model=model,
            max_tokens=300 if model == CHAT else 1600,
        )
        gen_prompt = usage.prompt_tokens - before.prompt_tokens
        gen_completion = usage.completion_tokens - before.completion_tokens
        gen_cached = usage.cached_tokens - before.cached_tokens
        gen_cost = cost_for(model, gen_prompt, gen_completion, gen_cached)

        judge_before = Usage(usage.prompt_tokens, usage.completion_tokens,
                             usage.cached_tokens, usage.calls)
        score, reason_text = judge(client, question, answer, usage)
        judge_prompt = usage.prompt_tokens - judge_before.prompt_tokens
        judge_completion = usage.completion_tokens - judge_before.completion_tokens
        judge_cached = usage.cached_tokens - judge_before.cached_tokens
        judge_cost = cost_for(CHAT, judge_prompt, judge_completion, judge_cached)

        keyword_cov, _ = coverage(answer, question)
        keyword_pass = keyword_cov >= 0.65
        judge_pass = score is not None and score >= 4
        final_pass = keyword_pass and judge_pass
        seconds = time.perf_counter() - question_started

        record = {
            "qid": question.id,
            "strategy": name,
            "model": model,
            "route_reason": reason,
            "question": question.question,
            "answer": answer,
            "keyword_coverage": keyword_cov,
            "keyword_pass": keyword_pass,
            "judge_score": score,
            "judge_pass": judge_pass,
            "final_pass": final_pass,
            "judge_reason": reason_text,
            "prompt_tokens": gen_prompt,
            "completion_tokens": gen_completion,
            "cached_tokens": gen_cached,
            "gen_cost": gen_cost,
            "judge_cost": judge_cost,
            "cost": gen_cost + judge_cost,
            "seconds": seconds,
        }
        records.append(record)
        mark = "✅" if final_pass else "❌"
        print(
            f"{question.id:<5}{model:<20}{keyword_cov:>5.2f}{score if score else '-':>7}"
            f"{mark:>6}{gen_prompt + gen_completion:>8}{seconds:>7.2f}"
        )

    gen_cost = sum(r["gen_cost"] for r in records)
    judge_cost = sum(r["judge_cost"] for r in records)
    summary = {
        "strategy": name,
        "questions": len(records),
        "keyword_pass": sum(r["keyword_pass"] for r in records),
        "judge_pass": sum(r["judge_pass"] for r in records),
        "final_pass": sum(r["final_pass"] for r in records),
        "avg_judge": (
            sum(r["judge_score"] for r in records if r["judge_score"]) /
            max(sum(1 for r in records if r["judge_score"]), 1)
        ),
        "gen_prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "gen_completion_tokens": sum(r["completion_tokens"] for r in records),
        "gen_cost": gen_cost,
        "judge_cost": judge_cost,
        "total_cost": gen_cost + judge_cost,
        "seconds": time.perf_counter() - started,
        "calls_by_model": calls_by_model,
        "premium_ids": [r["qid"] for r in records if r["model"] == PREMIUM],
        "records": records,
    }
    print(
        f"[汇总] 双通过 {summary['final_pass']}/{len(records)}"
        f" | 生成 ¥{gen_cost:.4f} + 判分 ¥{judge_cost:.4f}"
        f" = ¥{summary['total_cost']:.4f} | {summary['seconds']:.1f}s"
    )
    print(f"[模型] {calls_by_model}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="模型路由对比实验")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", type=Path, default=PROJECT / "stage3" / "index")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (
        RUNTIME / "routing.json"
        if not args.limit
        else RUNTIME / f"routing_limit{args.limit}.json"
    )

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    retriever = Retriever.load(args.index)
    contexts: dict[str, str] = {}
    for question in questions:
        hits = retriever.search(
            question.question, top_k=ROUTER_TOP_K, mode="vector", max_per_source=2
        )
        contexts[question.id] = format_context(hits)

    client = make_client()
    results = [
        run_strategy(client, questions, contexts, name="always_chat", fixed_model=CHAT),
        run_strategy(client, questions, contexts, name="always_premium",
                     fixed_model=PREMIUM),
        run_strategy(client, questions, contexts, name="routed", fixed_model=None),
    ]

    baseline = results[0]
    premium = results[1]
    routed = results[2]
    print("\n=== 路由对比 ===")
    header = (
        f"{'策略':<16}{'双通过':>8}{'kw':>7}{'judge':>7}{'平均judge':>10}"
        f"{'生成¥':>9}{'总¥':>9}{'秒':>7}  模型构成"
    )
    print(header)
    print("-" * len(header))
    for summary in results:
        print(
            f"{summary['strategy']:<16}{summary['final_pass']}/{summary['questions']:<6}"
            f"{summary['keyword_pass']:>7}{summary['judge_pass']:>7}"
            f"{summary['avg_judge']:>10.2f}{summary['gen_cost']:>9.4f}"
            f"{summary['total_cost']:>9.4f}{summary['seconds']:>7.1f}  "
            f"{summary['calls_by_model']}"
        )

    saved = 1 - routed["gen_cost"] / premium["gen_cost"]
    print(
        f"\n[结论] 相对 always_premium：生成成本降低 {saved:.0%}，"
        f"质量 {routed['final_pass']}/{routed['questions']} vs "
        f"{premium['final_pass']}/{premium['questions']}。"
    )
    print(
        f"       相对 always_chat：质量 {routed['final_pass']} vs "
        f"{baseline['final_pass']}，生成成本 "
        f"¥{routed['gen_cost']:.4f} vs ¥{baseline['gen_cost']:.4f}。"
    )
    print(f"        路由到 premium 的题：{routed['premium_ids']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "questions": len(questions),
                "results": [
                    {key: value for key, value in summary.items() if key != "records"}
                    for summary in results
                ],
                "records": [record for summary in results for record in summary["records"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[保存] {out}")


if __name__ == "__main__":
    main()
