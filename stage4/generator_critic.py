"""阶段 4 · 生成者—检查者双 agent，以及单 agent 基线。

两种方案解决同一个编码任务：
  单 agent    ：一次调用，模型直接给代码。
  双 agent    ：生成者写代码 → 隐藏测试打分 → 检查者审查并给问题清单
                → 生成者按问题修改，最多 3 轮。

判分不靠模型自评，而是把代码和隐藏测试放进子进程执行（见 code_tasks.py）。

运行单个任务：
    uv run python stage4/generator_critic.py --task merge_intervals
    uv run python stage4/generator_critic.py --task merge_intervals --multi-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(ROOT))

from code_tasks import TASKS, CodeTask, TestResult, extract_code, run_tests  # noqa: E402

load_dotenv()

MODEL = "deepseek-chat"
PRICE_INPUT = 2.0 / 1_000_000
PRICE_OUTPUT = 8.0 / 1_000_000
MAX_ROUNDS = 3


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        return self.prompt_tokens * PRICE_INPUT + self.completion_tokens * PRICE_OUTPUT


@dataclass
class RunResult:
    task: str
    strategy: str
    ok: bool
    rounds: int
    usage: Usage
    seconds: float
    code: str = ""
    detail: str = ""


def make_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def chat(
    client: OpenAI,
    system: str,
    user: str,
    usage: Usage,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response_usage = response.usage
    usage.add(
        response_usage.prompt_tokens if response_usage else 0,
        response_usage.completion_tokens if response_usage else 0,
    )
    return (response.choices[0].message.content or "").strip()


SINGLE_SYSTEM = """\
你是资深 Python 程序员。根据需求写出完整、可运行、只用标准库的代码。
只输出代码，不要写解释，不要写测试，不要输出 Markdown 围栏。"""

GENERATOR_SYSTEM = """\
你是资深 Python 程序员。根据需求和审查意见写出完整、可运行、只用标准库的代码。
只输出代码，不要写解释，不要写测试，不要输出 Markdown 围栏。
如果给了审查意见，逐条修正；不要只改表面。"""

CRITIC_SYSTEM = """\
你是严格的代码审查者。输入包含：需求、候选代码、隐藏测试的运行结果。
请检查：功能是否符合需求、边界条件、异常类型、是否修改输入、是否用了标准库以外的依赖。
输出 JSON（不要 Markdown 围栏）：
{"verdict": "approve" 或 "revise", "issues": ["可执行的具体修改建议", ...]}
规则：
1. 隐藏测试失败时必须 verdict=revise，并把失败原因写进 issues；
2. 隐藏测试全部通过时，只有代码明确违反需求、或存在测试没覆盖的正确性问题才 revise；
   不要因为风格、性能或需求没要求的扩展而 revise；
3. 最多 5 条 issues，每条都要让作者知道改哪里、怎么改。"""


def single_agent(client: OpenAI, task: CodeTask) -> RunResult:
    usage = Usage()
    started = time.perf_counter()
    code = extract_code(chat(client, SINGLE_SYSTEM, task.spec, usage, max_tokens=1200))
    test = run_tests(code, task)
    return RunResult(
        task=task.slug,
        strategy="single",
        ok=test.ok,
        rounds=1,
        usage=usage,
        seconds=time.perf_counter() - started,
        code=code,
        detail=test.brief(),
    )


def _parse_review(text: str) -> dict[str, Any]:
    """模型偶尔会带围栏或前后废话，尽力取出 JSON。"""
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"verdict": "revise", "issues": [f"审查输出不是合法 JSON：{text[:200]}"]}
    if not isinstance(data, dict):
        return {"verdict": "revise", "issues": ["审查输出必须是 JSON 对象"]}
    verdict = str(data.get("verdict", "revise")).lower()
    issues = data.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    return {"verdict": "approve" if verdict == "approve" else "revise",
            "issues": [str(item) for item in issues][:5]}


def multi_agent(client: OpenAI, task: CodeTask, *, verbose: bool = True) -> RunResult:
    usage = Usage()
    started = time.perf_counter()
    code = ""
    feedback = ""
    last_test = TestResult(False, "尚未运行")
    rounds = 0

    for round_number in range(1, MAX_ROUNDS + 1):
        rounds = round_number
        generator_input = task.spec
        if feedback:
            generator_input += f"\n\n上一轮的审查意见：\n{feedback}"
        code = extract_code(
            chat(client, GENERATOR_SYSTEM, generator_input, usage, max_tokens=1200)
        )
        last_test = run_tests(code, task)
        if verbose:
            print(f"  [生成者] 第 {round_number} 轮代码 {len(code)} 字符，"
                  f"测试 {'通过' if last_test.ok else '未通过'}", flush=True)

        critic_input = (
            f"需求：\n{task.spec}\n\n候选代码：\n{code}\n\n"
            f"隐藏测试结果：\n{last_test.brief()}"
        )
        review_text = chat(
            client, CRITIC_SYSTEM, critic_input, usage, max_tokens=600
        )
        review = _parse_review(review_text)
        if verbose:
            print(f"  [检查者] verdict={review['verdict']}，"
                  f"issues={len(review['issues'])} 条", flush=True)
            for issue in review["issues"]:
                print(f"           - {issue}", flush=True)

        if last_test.ok and review["verdict"] == "approve":
            break
        # 测试失败时，即使检查者说 approve 也不能算完成
        feedback = "\n".join(f"- {issue}" for issue in review["issues"])
        if not last_test.ok:
            feedback += f"\n- 隐藏测试输出：\n{last_test.brief(500)}"

    return RunResult(
        task=task.slug,
        strategy="multi",
        ok=last_test.ok,
        rounds=rounds,
        usage=usage,
        seconds=time.perf_counter() - started,
        code=code,
        detail=last_test.brief(),
    )


def print_result(result: RunResult) -> None:
    print(
        f"\n[{result.strategy}] {result.task}: "
        f"{'✅ 通过' if result.ok else '❌ 未通过'} | "
        f"{result.rounds} 轮 | {result.usage.calls} 次模型调用 | "
        f"{result.usage.total} tok | 约 ¥{result.usage.cost:.4f} | "
        f"{result.seconds:.1f}s"
    )
    if not result.ok:
        print(f"  测试输出：{result.detail[:500]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 4 · 生成者—检查者双 agent")
    parser.add_argument("--task", default="merge_intervals",
                        choices=[task.slug for task in TASKS])
    parser.add_argument("--multi-only", action="store_true")
    parser.add_argument("--single-only", action="store_true")
    args = parser.parse_args()

    task = next(item for item in TASKS if item.slug == args.task)
    client = make_client()
    print(f"任务：{task.slug}\n需求：\n{task.spec}")

    if not args.multi_only:
        print("\n=== 单 agent ===")
        print_result(single_agent(client, task))
    if not args.single_only:
        print("\n=== 生成者—检查者双 agent ===")
        print_result(multi_agent(client, task))


if __name__ == "__main__":
    main()
