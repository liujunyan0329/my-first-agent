"""阶段 4 · 并行工具调用：同一条模型回复里的多个调用，串行跑 vs 并行跑。

关键点：
  1. 模型本来就会一次返回多个 tool_calls——要不要并发执行，是你的决定；
  2. 只有 I/O 密集型工具（网络、磁盘、子进程）并行才有收益；
  3. asyncio 的事件循环跑在单线程里，同步工具必须用 asyncio.to_thread 包一层。

运行：
    uv run python stage4/parallel_tools.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools import execute_tool, tool_schemas  # noqa: E402

load_dotenv()
MODEL = "deepseek-chat"
PRICE_INPUT = 2.0 / 1_000_000
PRICE_OUTPUT = 8.0 / 1_000_000

SYSTEM_PROMPT = """\
你是一个会使用工具的助手。
当用户要求查询多个相互独立的对象时，请在同一条回复里同时发起多个工具调用，
不要一个查完再查下一个。
不需要复述工具结果，模型这一轮只负责发起调用。"""


def ask_for_calls(client: OpenAI, prompt: str) -> tuple[list[Any], int, int]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tools=tool_schemas(),
        temperature=0.0,
    )
    message = response.choices[0].message
    calls = [c for c in message.tool_calls or [] if c.type == "function"]
    usage = response.usage
    return (
        calls,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def run_serial(calls: list[Any]) -> list[str]:
    return [
        execute_tool(call.function.name, call.function.arguments) for call in calls
    ]


async def run_parallel(calls: list[Any]) -> list[str]:
    """execute_tool 是同步函数；to_thread 把它丢进线程池，gather 等所有结果。"""

    async def one(call: Any) -> str:
        return await asyncio.to_thread(
            execute_tool, call.function.name, call.function.arguments
        )

    return list(await asyncio.gather(*(one(call) for call in calls)))


def bench(label: str, calls: list[Any], prompt_tokens: int, completion_tokens: int) -> None:
    print(f"\n=== {label}：{len(calls)} 个工具调用 ===")
    for call in calls:
        print(f"  - {call.function.name}({call.function.arguments})")

    started = time.perf_counter()
    serial_results = run_serial(calls)
    serial = time.perf_counter() - started

    started = time.perf_counter()
    parallel_results = asyncio.run(run_parallel(calls))
    parallel = time.perf_counter() - started

    same = serial_results == parallel_results
    speedup = serial / parallel if parallel else float("inf")
    cost = prompt_tokens * PRICE_INPUT + completion_tokens * PRICE_OUTPUT
    print(f"\n  {'模式':<8}{'耗时':>8}   说明")
    print(f"  {'串行':<8}{serial:>7.2f}s   一个接一个执行")
    print(f"  {'并行':<8}{parallel:>7.2f}s   asyncio.gather")
    print(f"  加速比：{speedup:.1f}×；结果一致：{'是' if same else '否'}")
    print(f"  模型调用：{prompt_tokens + completion_tokens} tok，约 ¥{cost:.4f}")
    print("\n  并行结果：")
    for result in parallel_results:
        print(f"    {result}")


def main() -> None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    weather_calls, p1, c1 = ask_for_calls(
        client, "请同时查询北京、上海、东京、纽约四个城市的天气。"
    )
    if weather_calls:
        bench("模拟网络接口（每次 0.8s）", weather_calls, p1, c1)
    else:
        print("模型没有发起工具调用，换一次提示再试。")

    time_calls, p2, c2 = ask_for_calls(
        client, "请同时查询北京、上海、东京、纽约四个城市现在的时间。"
    )
    if time_calls:
        bench("本地计算（几乎不耗时）", time_calls, p2, c2)

    print("\n结论：并发只对 I/O 等待有效。纯 CPU 计算放进线程池反而会因为 GIL 没有收益。")


if __name__ == "__main__":
    main()
