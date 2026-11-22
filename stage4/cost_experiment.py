"""阶段 4 · 成本实验：同一批任务，单 agent vs 生成者—检查者双 agent。

记录四个指标：成功率、模型调用次数、token、费用。
判分器是隐藏测试（code_tasks.py），不是模型自评。

运行：
    uv run python stage4/cost_experiment.py
    uv run python stage4/cost_experiment.py --tasks merge_intervals parse_duration
    uv run python stage4/cost_experiment.py --repeat 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from code_tasks import TASKS  # noqa: E402
from generator_critic import (  # noqa: E402
    RunResult,
    Usage,
    make_client,
    multi_agent,
    single_agent,
)


def summarize(results: list[RunResult]) -> dict[str, float | int]:
    if not results:
        return {}
    usage = Usage()
    for result in results:
        usage.prompt_tokens += result.usage.prompt_tokens
        usage.completion_tokens += result.usage.completion_tokens
        usage.calls += result.usage.calls
    passed = sum(1 for result in results if result.ok)
    return {
        "passed": passed,
        "total": len(results),
        "rate": passed / len(results),
        "calls": usage.calls,
        "tokens": usage.total,
        "cost": usage.cost,
        "seconds": sum(result.seconds for result in results),
    }


def print_table(results: list[RunResult]) -> None:
    print("\n=== 逐题结果（多次重复时取平均与通过率）===")
    print(
        f"{'任务':<20}{'单agent':<12}{'双agent':<12}"
        f"{'单tok':>8}{'双tok':>8}{'单¥':>9}{'双¥':>9}"
    )
    print("-" * 86)
    slugs = list(dict.fromkeys(r.task for r in results))
    for slug in slugs:
        cells: dict[str, str] = {}
        tokens: dict[str, float] = {}
        costs: dict[str, float] = {}
        for strategy in ("single", "multi"):
            group = [r for r in results if r.task == slug and r.strategy == strategy]
            if not group:
                cells[strategy], tokens[strategy], costs[strategy] = "-", 0, 0.0
                continue
            passed = sum(1 for r in group if r.ok)
            cells[strategy] = f"{passed}/{len(group)}"
            tokens[strategy] = sum(r.usage.total for r in group) / len(group)
            costs[strategy] = sum(r.usage.cost for r in group) / len(group)
        print(
            f"{slug:<20}{cells['single']:<12}{cells['multi']:<12}"
            f"{tokens['single']:>8.0f}{tokens['multi']:>8.0f}"
            f"{costs['single']:>9.4f}{costs['multi']:>9.4f}"
        )

    print("\n=== 汇总 ===")
    for strategy in ("single", "multi"):
        stats = summarize([r for r in results if r.strategy == strategy])
        print(
            f"{strategy:<8} 成功率 {stats['passed']}/{stats['total']} "
            f"({stats['rate']:.0%}) | 调用 {stats['calls']} 次 | "
            f"{stats['tokens']} tok | 约 ¥{stats['cost']:.4f} | "
            f"{stats['seconds']:.1f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 4 · 单/双 agent 成本实验")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="任务 slug；默认全部")
    parser.add_argument("--repeat", type=int, default=1, help="每个任务重复几次")
    args = parser.parse_args()

    selected = (
        TASKS if not args.tasks
        else [task for task in TASKS if task.slug in set(args.tasks)]
    )
    if not selected:
        raise SystemExit("没有匹配的任务")

    client = make_client()
    results: list[RunResult] = []
    started = time.perf_counter()

    for repeat in range(1, args.repeat + 1):
        for task in selected:
            print(f"\n[{repeat}/{args.repeat}] 任务 {task.slug}")
            print("  --- 单 agent ---", flush=True)
            single = single_agent(client, task)
            print(f"  {'✅' if single.ok else '❌'} "
                  f"{single.usage.total} tok，{single.usage.calls} 次调用", flush=True)
            results.append(single)

            print("  --- 双 agent ---", flush=True)
            multi = multi_agent(client, task, verbose=False)
            print(f"  {'✅' if multi.ok else '❌'} "
                  f"{multi.usage.total} tok，{multi.usage.calls} 次调用，"
                  f"{multi.rounds} 轮", flush=True)
            results.append(multi)

    print_table(results)
    total_seconds = time.perf_counter() - started

    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    report = {
        "model": "deepseek-chat",
        "repeat": args.repeat,
        "results": [
            {
                "task": r.task, "strategy": r.strategy, "ok": r.ok,
                "rounds": r.rounds, "calls": r.usage.calls,
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
                "cost": r.usage.cost, "seconds": r.seconds,
            }
            for r in results
        ],
    }
    output = runtime / "cost_experiment.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {total_seconds:.1f}s；结果已保存到 {output}")
    print(
        "\n怎么读：双 agent 的调用次数和 token 通常明显更高。"
        "\n只有当它把成功率提上来时，多花的钱才值；简单任务上它基本是纯开销。"
    )


if __name__ == "__main__":
    main()
