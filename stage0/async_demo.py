"""任务④进阶：async / await —— 阶段 0 唯一会真正卡住你的地方。

运行：uv run python stage0/async_demo.py

三个实验，一次跑完：
  实验 1  串行 vs 并发：为什么 asyncio.gather 能把 2 次请求的耗时压成 1 次
  实验 2  异步流式输出：async for
  实验 3  阻塞实验：time.sleep 会把整个事件循环卡死
"""

import asyncio
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

QUESTIONS = [
    "用一句话解释什么是向量数据库。",
    "用一句话解释什么是 ReAct。",
    "用一句话解释什么是 RAG。",
]


async def ask(question: str) -> str:
    """一次完整（非流式）请求。注意：函数用 async def 定义，返回的是协程。"""
    resp = await client.chat.completions.create(  # await = 在这里让出控制权
        model="deepseek-chat",
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


async def experiment_serial_vs_concurrent() -> None:
    print("=" * 60)
    print("实验 1：串行 vs 并发")
    print("=" * 60)

    # 串行：一个做完再做下一个，总耗时 ≈ 3 次之和
    t0 = time.perf_counter()
    for q in QUESTIONS:
        await ask(q)
    serial = time.perf_counter() - t0
    print(f"串行 for + await      : {serial:.2f} s")

    # 并发：三个请求同时在飞，总耗时 ≈ 最慢的那一个
    t0 = time.perf_counter()
    results = await asyncio.gather(*(ask(q) for q in QUESTIONS))
    concurrent = time.perf_counter() - t0
    print(f"asyncio.gather 并发   : {concurrent:.2f} s   ← 快 {serial / concurrent:.1f} 倍")
    for q, a in zip(QUESTIONS, results, strict=True):
        print(f"  · {q}\n    → {a.strip()}")


async def experiment_async_stream() -> None:
    print("\n" + "=" * 60)
    print("实验 2：异步流式输出（async for）")
    print("=" * 60)

    t0 = time.perf_counter()
    first: float | None = None

    stream = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "用 150 字解释什么是 Agent 的上下文窗口。"}],
        temperature=0.3,
        stream=True,
        stream_options={"include_usage": True},
    )

    print("AI  > ", end="", flush=True)
    usage = None
    async for chunk in stream:  # 注意是 async for，不是 for
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            if first is None:
                first = time.perf_counter()
            print(delta, end="", flush=True)

    print(f"\n首字延迟 {first - t0:.2f} s" if first else "")
    if usage is not None:
        print(f"token: 输入 {usage.prompt_tokens} / 输出 {usage.completion_tokens}")


async def experiment_blocking() -> None:
    print("\n" + "=" * 60)
    print("实验 3：阻塞事件循环（把 time.sleep 换成 asyncio.sleep 再看一次）")
    print("=" * 60)

    async def ticker(name: str, n: int) -> None:
        for i in range(n):
            print(f"  [{name}] 第 {i + 1} 次心跳")
            time.sleep(0.5)  # ← 罪魁祸首。改成 await asyncio.sleep(0.5) 就正常了

    t0 = time.perf_counter()
    await asyncio.gather(ticker("A", 3), ticker("B", 3))
    print(f"总耗时 {time.perf_counter() - t0:.2f} s")
    print("  如果 A、B 的心跳是交替出现的 → 并发正常")
    print("  如果 A 连打 3 次才轮到 B   → 事件循环被阻塞了")


async def main() -> None:
    await experiment_serial_vs_concurrent()
    await experiment_async_stream()
    await experiment_blocking()


if __name__ == "__main__":
    # 唯一的入口：启动事件循环，把协程塞进去跑
    asyncio.run(main())
