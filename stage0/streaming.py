"""任务④：流式输出（同步版）+ 测量首字延迟。

运行：uv run python stage0/streaming.py
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

QUESTION = "用 200 字左右解释什么是 AI Agent。"


def main() -> None:
    start = time.perf_counter()
    first_token_at: float | None = None
    usage = None

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": QUESTION},
        ],
        temperature=0.3,
        stream=True,
        # 让服务端在最后一个 chunk 里带上 token 用量。
        # 如果你的服务商不支持这个参数，删掉这一行，usage 就取不到了。
        stream_options={"include_usage": True},
    )

    print("AI  > ", end="", flush=True)

    for chunk in stream:
        # 收尾的那个 chunk 带 usage，但 choices 是空列表，必须先判断
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue

        # 流式返回的是 delta（增量），不是 message（整段）
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(delta, end="", flush=True)

    total = time.perf_counter() - start
    print("\n" + "-" * 50)
    if first_token_at is not None:
        print(f"首字延迟 TTFT : {first_token_at - start:.2f} s   ← 用户真正感知到的等待")
    print(f"总耗时        : {total:.2f} s")
    if usage is not None:
        print(f"token         : 输入 {usage.prompt_tokens} / 输出 {usage.completion_tokens}")


if __name__ == "__main__":
    main()
