"""阶段 1 · 任务②③⑤⑦：把单次调用改成 while 循环，接 3 个工具。

这个文件故意保持“平铺”写法（没有类、没有注册表、没有预算），
只为了让你看清核心循环长什么样。下一步 agent.py 再把它工程化。

运行：
  uv run python stage1/s2_loop.py
  uv run python stage1/s2_loop.py "北京现在几点？写进 s2_result.txt"
"""

import ast
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

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

MODEL = "deepseek-chat"
MAX_TURNS = 8
BASE_DIR = Path(__file__).resolve().parent

CITY_OFFSETS = {"北京": 8, "东京": 9, "伦敦": 0, "纽约": -5}


# ---------------- 工具实现 ----------------
def get_current_time(city: str) -> str:
    offset = CITY_OFFSETS.get(city.strip())
    if offset is None:
        return f"错误：不认识城市 {city!r}。支持：{', '.join(CITY_OFFSETS)}"
    now = datetime.now(timezone(timedelta(hours=offset)))
    return f"{city}现在是 {now:%Y-%m-%d %H:%M:%S}（UTC{offset:+d}）"


def calculate(expression: str) -> str:
    """只允许数字和四则运算。不要把裸 eval 暴露给模型！"""
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
               ast.Pow, ast.USub, ast.UAdd)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return f"错误：无法解析 {expression!r}"
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return "错误：表达式里有不支持的语法"
    try:
        return f"{expression} = {eval(compile(tree, '<calc>', 'eval'), {'__builtins__': {}}, {})}"
    except ZeroDivisionError:
        return "错误：除数不能为 0"


def write_file(path: str, content: str) -> str:
    target = (BASE_DIR / path).resolve()
    if not target.is_relative_to(BASE_DIR):
        return "错误：只能写工作目录内的文件"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {path}（{len(content)} 字符）"


# ---------------- 工具描述（发给模型） ----------------
TOOLS: list[ChatCompletionToolParam] = [
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "查询城市当前本地时间。问“现在几点”时使用，不要猜。",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "城市中文名"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "计算算术表达式，如 9-8。做数字运算时使用，不要心算。",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string", "description": "算式"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "把文本写入工作目录内的文件。用户要求保存结果时使用。",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "相对路径"},
                                      "content": {"type": "string", "description": "完整文本"}},
                       "required": ["path", "content"]}}},
]

FUNCS: dict[str, Callable[..., str]] = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "write_file": write_file,
}


def execute(name: str, arguments: str) -> str:
    """执行工具：所有异常都变成字符串返回，agent 不崩。"""
    func = FUNCS.get(name)
    if func is None:
        return f"错误：没有名为 {name!r} 的工具。可用：{', '.join(FUNCS)}"
    try:
        args = json.loads(arguments or "{}")
        return func(**args)
    except Exception as exc:
        return f"错误：工具 {name} 执行失败：{type(exc).__name__}: {exc}"


# ---------------- 核心循环 ----------------
def run_agent(question: str) -> str:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "你是会使用工具的助手。需要事实就调用工具，不要编造。"},
        {"role": "user", "content": question},
    ]

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- 第 {turn} 轮 ---")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        message = resp.choices[0].message
        calls = [c for c in message.tool_calls or [] if c.type == "function"]

        # 关键：assistant 的原始输出（含 tool_calls）必须先回填历史
        assistant_msg: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": message.content or "",
        }
        if calls:
            assistant_msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ]
        messages.append(assistant_msg)

        if not calls:                    # 停止条件 1：模型不再调用工具
            print(f"[最终回答] {message.content or ''}")
            return message.content or ""

        for call in calls:
            print(f"[调用] {call.function.name}({call.function.arguments})")
            result = execute(call.function.name, call.function.arguments)
            print(f"[结果] {result[:120]}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )
        if resp.usage is not None:
            print(f"[用量] 输入 {resp.usage.prompt_tokens} / 输出 {resp.usage.completion_tokens}")

    # 停止条件 2：达到最大轮次
    print(f"[已停止] 达到最大轮次 {MAX_TURNS}")
    return ""


def main() -> None:
    question = " ".join(sys.argv[1:]) or "北京现在几点？把结果写进 s2_result.txt"
    started = time.perf_counter()
    run_agent(question)
    print(f"\n总耗时 {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
