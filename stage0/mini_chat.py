"""任务⑤：迷你多轮对话（异步流式 + 滑动窗口）

综合运用前面几个任务的全部技能：
  · 手动维护 messages 列表（多轮上下文）
  · AsyncOpenAI + async for 流式输出，并测量首字延迟 TTFT
  · 每轮打印历史条数 / 本轮 token / 累计 token
  · 累计输入 token 超过预算时，丢掉最早的一轮（滑动窗口）

运行：uv run python stage0/mini_chat.py
退出：输入 exit / quit / q，或按 Ctrl+C
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

# ---------------- 可调参数 ----------------

MODEL = "deepseek-chat"
TEMPERATURE = 0.3

# 触发滑窗的阈值：累计输入 token 超过它，就丢掉最早的一轮对话。
# 想快速看到效果可以临时改成 200。
INPUT_TOKEN_BUDGET = 3000

SYSTEM_PROMPT = "你是一个简洁的助手，回答控制在 100 字以内。"

EXIT_WORDS = {"exit", "quit", "q"}

# ----------------------------------------


def rule(char: str = "-", width: int = 56) -> None:
    """打印一条分隔线。"""
    print(char * width)


def rounds_of(messages: list[dict]) -> int:
    """历史里有几轮对话。messages[0] 是 system，所以先减掉它再除以 2。"""
    return (len(messages) - 1) // 2


def drop_oldest_round(messages: list[dict]) -> tuple[str, str] | None:
    """丢掉最早的一轮（user + assistant），返回被丢掉的内容用于报告。

    system 提示词永远保留在 messages[0]，不会被丢。
    只剩一轮时返回 None（再丢就没有上下文了，留着更划算）。
    """
    if rounds_of(messages) < 2:
        return None

    user_msg = messages.pop(1)
    assistant_msg = messages.pop(1)
    return user_msg["content"], assistant_msg["content"]


async def chat_once(messages: list[dict], user_input: str):
    """发一轮请求：流式打印回答，把 assistant 回复追加进 messages。

    返回 (reply, usage, ttft)：
      失败时 reply 为 None，调用方应跳过本轮统计。
    """
    messages.append({"role": "user", "content": user_input})

    t0 = time.perf_counter()
    first_at: float | None = None
    parts: list[str] = []
    usage = None

    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            stream=True,
            stream_options={"include_usage": True},
        )

        print("AI  > ", end="", flush=True)
        async for chunk in stream:
            # 收尾那个块带 usage，但 choices 是空列表，必须先判断
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content
            if delta:
                if first_at is None:
                    first_at = time.perf_counter()
                parts.append(delta)
                print(delta, end="", flush=True)
        print()

    except Exception as exc:
        # 请求死在半路：把刚才那条 user 消息退掉，否则历史会出现
        # "user 后面没有 assistant" 的残缺结构，下一轮请求可能被拒。
        messages.pop()
        print(f"\n[请求失败] {type(exc).__name__}: {exc}")
        return None, None, None

    reply = "".join(parts)
    messages.append({"role": "assistant", "content": reply})

    ttft = (first_at - t0) if first_at is not None else None
    return reply, usage, ttft


async def main() -> None:
    # messages 就是全部的对话状态：一个列表，我们要自己维护它。
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    total_in = 0
    total_out = 0
    turns = 0

    rule("=")
    print(f"mini_chat · {MODEL} · 流式 · 滑窗阈值 {INPUT_TOKEN_BUDGET} 输入 token")
    print("输入 exit / quit / q 退出，Ctrl+C 随时中断")
    rule("=")

    while True:
        # input() 是阻塞调用，直接写会卡死事件循环（见 async_demo 实验 3），
        # 所以丢到线程池里跑，主循环该让出的控制权还是要让。
        try:
            prompt = f"\n你({rounds_of(messages)}轮) > "
            user_input = (await asyncio.to_thread(input, prompt)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_WORDS:
            break

        reply, usage, ttft = await chat_once(messages, user_input)
        if reply is None:
            continue  # 这一轮失败了，不计入统计

        turns += 1
        turn_in = usage.prompt_tokens if usage is not None else None
        turn_out = usage.completion_tokens if usage is not None else None
        if turn_in is not None:
            total_in += turn_in
        if turn_out is not None:
            total_out += turn_out

        # 本轮结束后立刻裁剪，下一轮请求用的就是裁剪过的历史。
        # 注意这里用的是「累计」输入 token：一旦超过预算，之后每轮都会丢一轮，
        # 窗口长度就稳定在第一次触发时的规模，不再增长。
        trimmed = None
        if total_in > INPUT_TOKEN_BUDGET:
            trimmed = drop_oldest_round(messages)

        print()
        rule()
        print(f"历史条数   : {len(messages)} 条（system 1 + 对话 {rounds_of(messages)} 轮）")
        if usage is None:
            print("本轮 token : 输入 N/A / 输出 N/A（服务商没返回 usage）")
            print("累计 token : 输入 N/A / 输出 N/A")
        else:
            print(f"本轮 token : 输入 {turn_in} / 输出 {turn_out}")
            print(
                f"累计 token : 输入 {total_in} / 输出 {total_out}"
                f"（合计 {total_in + total_out}）"
            )
        print(f"首字延迟   : {'N/A' if ttft is None else f'{ttft:.2f} s'}")
        rule()
        if total_in > INPUT_TOKEN_BUDGET:
            if trimmed is not None:
                old_user, _ = trimmed
                snippet = old_user if len(old_user) <= 16 else old_user[:16] + "…"
                print(
                    f"[滑窗] 累计输入 {total_in} > {INPUT_TOKEN_BUDGET}，"
                    f"丢掉最早一轮「{snippet}」"
                )
            else:
                print(
                    f"[滑窗] 累计输入 {total_in} > {INPUT_TOKEN_BUDGET}，"
                    "但只剩一轮上下文，保留不丢"
                )

    rule("=")
    print("会话结束")
    print(f"轮数       : {turns}")
    print(f"累计 token : 输入 {total_in} / 输出 {total_out}（合计 {total_in + total_out}）")
    print(f"最终历史   : {len(messages)} 条（对话 {rounds_of(messages)} 轮）")
    rule("=")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n(已中断退出)")
