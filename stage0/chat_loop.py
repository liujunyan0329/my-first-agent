"""任务③：多轮对话 —— 手动维护 messages 列表。

运行：uv run python stage0/chat_loop.py
退出：输入 exit

核心认知：LLM API 是无状态的。它不记得上一轮说了什么，
所谓"记忆"就是每次请求把【整个历史】重新发一遍。
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

client = OpenAI(
    api_key = os.environ["DEEPSEEK_API_KEY"],
    base_url = "https://api.deepseek.com",
)

SYSTEM_PROMPT = "你是一个智能简洁助手，每段回答控制在两句话以内。"

def main() ->None:
    messages:list[ChatCompletionMessageParam] = [
        {"role":"system","content":SYSTEM_PROMPT},
    ]

    print("多轮对话开始，输入exit退出。")

    while True:
        user_input = input("\n你 >").strip()
        if user_input in {"exit","quit"}:
            break
        if not user_input:
            continue

        messages.append({"role":"user","content":user_input})

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
        )

        reply = resp.choices[0].message.content or""

        messages.append({"role":"assistant","content":reply})

        usage = resp.usage
        print(f"\nAI > {reply}")
        print(
            f"[历史{len(messages)}条信息 |"
            f"输入tokens:{usage.prompt_tokens if usage else '?'} |"
            f"输出tokens:{usage.completion_tokens if usage else '?'}]")
if __name__ == "__main__":
    main()




