"""任务⑤：超 context window / 超参数上限，看服务端报什么错。

运行：
  uv run python stage0/overflow.py           # 默认：免费触发一次 400 + 便宜的长输入探测
  uv run python stage0/overflow.py 200000    # 指定长输入的字符数（会按输入 token 计费！）

分三步：
  A. 免费看一次 400 —— max_tokens 超过服务端允许的上限
  B. 长输入探测 —— 观察 token/字 的实际比值（顺便看会不会撞上窗口上限）
  C. 把测出来的比值变成预算函数 —— 这才是这个实验真正要带走的东西

注意：被拒绝（400）的请求通常不计费，但成功的请求按输入 token 计费，
所以别拿超大文本反复刷。
"""

import os
import sys

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

FILLER_UNIT = "这是一段用来占满上下文的中文文本。"
DEFAULT_CHARS = 30_000  # 约 1.8 万 token，费用可以忽略

# 实测出来的比值（跑完实验 B 你会自己得到这个数），用于估算：
TOKENS_PER_CHAR = 0.59


def estimate_tokens(text: str) -> int:
    """本地粗略估算 token 数。不需要精确，能提前挡住爆窗就够了。"""
    return int(len(text) * TOKENS_PER_CHAR)


def experiment_a_free_400() -> None:
    print("=" * 64)
    print("实验 A：免费看一次 400（max_tokens 超过服务端上限）")
    print("=" * 64)
    try:
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=999_999,  # 明显超过上限
        )
        print("居然没报错？那说明这个上限被放开了。")
    except BadRequestError as e:
        print(f"status_code : {e.status_code}")
        print(f"type        : invalid_request_error")
        print(f"code        : {e.code}")
        print(f"message     : {e.message}")
        print("\n--- 原始响应体（就是一段 JSON）---")
        print(e.response.text)
        print("\n要记住的：")
        print("· 400 invalid_request_error = 【请求阶段】就被拒了，模型根本没开始生成")
        print("· 消息里直接告诉了你合法区间，照着改就行")


def experiment_b_long_input(n_chars: int) -> None:
    print("\n" + "=" * 64)
    print(f"实验 B：长输入探测（{n_chars:,} 个字符）")
    print("=" * 64)

    filler = FILLER_UNIT * max(1, n_chars // len(FILLER_UNIT))
    print(f"本地估算 ≈ {estimate_tokens(filler):,} token")

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": filler}],
            max_tokens=1,  # 只关心输入能不能进得去
        )
        usage = resp.usage
        n = usage.prompt_tokens if usage else 0
        print(f"服务端接受。prompt_tokens = {n:,}")
        print(f"实测比值 = {n / len(filler):.2f} token/字   ← 中文大致就是 0.6，记下来")
        print("还没撞到墙。想真撞一次就把字符数往上翻（比如 2000000），")
        print("但那要花钱，而且本机这个模型的窗口非常大。")
    except BadRequestError as e:
        print("=== 撞上了！服务端返回 400 ===")
        print(f"message : {e.message}")
        print(e.response.text)


def experiment_c_budget(filler: str) -> None:
    print("\n" + "=" * 64)
    print("实验 C：把比值变成预算（这才是真正要带走的东西）")
    print("=" * 64)
    budget = 32_000
    print(f"假设我给自己定 {budget:,} token 的预算：")
    print(f"  我能塞进去的中文正文 ≈ {int(budget / TOKENS_PER_CHAR):,} 个字")
    print(f"  当前这段文本估算 {estimate_tokens(filler):,} token "
          f"→ {'超预算，先摘要/截断再发' if estimate_tokens(filler) > budget else '放心发'}")
    print("\n为什么不用等 API 报错？因为报错说明这一轮已经废了：")
    print("  用户等了一轮、你还得重试、而且 agent 的多步推理整个断掉。")
    print("  阶段 3 的上下文工程，本质就是在发请求【之前】做这件事。")


def main() -> None:
    n_chars = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHARS
    experiment_a_free_400()
    experiment_b_long_input(n_chars)
    experiment_c_budget(FILLER_UNIT * (n_chars // len(FILLER_UNIT)))


if __name__ == "__main__":
    main()
