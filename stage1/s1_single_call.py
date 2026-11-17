"""阶段 1 · 任务①：单次工具调用 —— 手动把 tool calling 协议走一遍。

流程：
  你发问题 + 工具清单
  → 模型返回的不是文本，而是 {"name": ..., "arguments": ...}
  → 你执行真正的函数
  → 把结果作为 tool 消息塞回历史
  → 再发一次，模型给出最终答案

运行：uv run python stage1/s1_single_call.py
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY") or "",
    base_url="https://api.deepseek.com",
)

# 假装有一个天气数据库。真实的工具可以去调气象 API。
WEATHER_DB = {"北京": "晴，18°C，北风 2 级", "上海": "多云，22°C", "东京": "小雨，16°C"}


def get_weather(city: str) -> str:
    if city not in WEATHER_DB:
        return f"错误：没有 {city} 的天气数据。可用城市：{', '.join(WEATHER_DB)}"
    return f"{city}天气：{WEATHER_DB[city]}"


TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询一个城市当前的天气。用户问天气时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京"}
                },
                "required": ["city"],
            },
        },
    }
]


def main() -> None:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "你是一个会使用工具的助手。"},
        {"role": "user", "content": "北京天气怎么样？"},
    ]

    # ---------- 第 1 次请求：模型只返回“我要调用谁” ----------
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )
    message = resp.choices[0].message
    calls = [c for c in message.tool_calls or [] if c.type == "function"]

    print("模型的 content :", repr(message.content))
    print("finish_reason  :", resp.choices[0].finish_reason)
    for call in calls:
        print(f"模型要求调用   : {call.function.name}({call.function.arguments})")

    if not calls:
        print("模型没要求调用工具，直接回答：", message.content)
        return

    # ---------- 把 assistant 的原始调用请求 append 回历史 ----------
    assistant_msg: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ],
    }
    messages.append(assistant_msg)

    # ---------- 你负责执行，并把结果塞回去 ----------
    for call in calls:
        args = json.loads(call.function.arguments)
        result = get_weather(**args)          # 真正执行的是你的代码
        print(f"本地执行结果   : {result}")
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": result}
        )

    # ---------- 第 2 次请求：模型看到结果后给最终答案 ----------
    resp2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )
    print("最终回答       :", resp2.choices[0].message.content)
    if resp2.usage is not None:
        print(
            f"token          : 输入 {resp2.usage.prompt_tokens} / "
            f"输出 {resp2.usage.completion_tokens}"
        )


if __name__ == "__main__":
    main()
