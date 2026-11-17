"""阶段 1 验收：不依赖任何 agent 框架、300 行以内的 agent。
Agent = LLM + 工具 + 循环 + 停止条件

用法：
  uv run python stage1/agent.py "北京现在几点？比东京早还是晚几小时？把结论写进 result.txt"
  uv run python stage1/agent.py            # 交互模式（历史跨轮保留）
"""
import ast
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

load_dotenv()

MODEL = "deepseek-chat"
WORKSPACE = Path(__file__).resolve().parent  # 文件工具只能在这个目录里活动

# ---------------- 停止条件（缺一个都可能无限循环烧钱） ----------------
MAX_TURNS = 12          # 最多调用模型 12 次
MAX_TOKENS = 40_000     # 输入 + 输出累计超过就停
MAX_SECONDS = 120.0     # 单次任务最多跑 2 分钟

SYSTEM_PROMPT = """\
你是一个会使用工具的助手。规则：
1. 不知道的事实必须查：时间用 get_current_time，算术用 calculate，
   文件内容用 read_file / list_files，保存结论用 write_file。不要凭空编造。
2. 一条消息里可以同时发起多个工具调用（比如一次查两个城市的时间）。
3. 工具返回以“错误：”开头时，先读错误信息，修正参数后重试。
4. 完成用户要求后，用简洁的中文给出最终结论，不要再调用工具。"""

# ------------- 工具 1：查时间（固定时差，不处理夏令时） -------------
CITY_OFFSETS: dict[str, int] = {
    "北京": 8, "上海": 8, "东京": 9, "首尔": 9, "悉尼": 11,
    "伦敦": 0, "巴黎": 1, "柏林": 1, "纽约": -5, "洛杉矶": -8,
}


def get_current_time(city: str) -> str:
    """返回某城市当前的本地时间和 UTC 偏移。"""
    city = city.strip()
    offset = CITY_OFFSETS.get(city)
    if offset is None:
        return f"错误：不认识城市 {city!r}。支持：{', '.join(CITY_OFFSETS)}"
    now = datetime.now(timezone(timedelta(hours=offset)))
    sign = "+" if offset >= 0 else "-"
    return f"{city}现在是 {now:%Y-%m-%d %H:%M:%S}（UTC{sign}{abs(offset)}）"


# ------------- 工具 2：算数（AST 白名单，不用裸 eval） -------------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)


def calculate(expression: str) -> str:
    """计算只含数字和 + - * / // % ** 的算式。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return f"错误：无法解析 {expression!r}。只支持数字、括号和 + - * / // % **。"
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return f"错误：表达式里有不支持的语法 {type(node).__name__}。"
    try:
        value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return "错误：除数不能为 0。请检查算式。"
    except Exception as exc:  # 比如 2 ** 999999 之类的极端情况
        return f"错误：计算失败：{type(exc).__name__}: {exc}"
    return f"{expression} = {value}"


# ------------- 工具 3~5：文件（限制在 WORKSPACE 内） -------------
def _resolve(path: str) -> Path:
    """把相对路径解析到工作目录内；越界就拒绝。"""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"路径越界：{path!r} 不在工作目录 {WORKSPACE.name} 内")
    return target


def read_file(path: str) -> str:
    """读取工作目录内的文本文件。"""
    target = _resolve(path)
    if not target.is_file():
        return f"错误：文件不存在：{path!r}。可以先用 list_files 看看有什么。"
    text = target.read_text(encoding="utf-8")
    if len(text) > 4000:
        return text[:4000] + f"\n…（已截断，原文共 {len(text)} 字符）"
    return text


def write_file(path: str, content: str) -> str:
    """把文本写入工作目录内的文件（会覆盖同名文件）。"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {target.relative_to(WORKSPACE)}（{len(content)} 字符）"


def list_files(directory: str = ".") -> str:
    """列出工作目录内某个目录的文件和子目录。"""
    target = _resolve(directory)
    if not target.is_dir():
        return f"错误：目录不存在：{directory!r}"
    entries = sorted(target.iterdir())[:50]
    lines = [("目录 " if p.is_dir() else "文件 ") + p.name for p in entries]
    return "\n".join(lines) if lines else "（空目录）"


# ------------- 工具注册表：名字 / 描述 / 参数 schema -------------
@dataclass(frozen=True)
class Tool:
    fn: Callable[..., str]
    description: str
    parameters: dict[str, Any]

    @property
    def name(self) -> str:
        return self.fn.__name__

    def schema(self) -> ChatCompletionToolParam:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


TOOLS: list[Tool] = [
    Tool(get_current_time,
         "查询一个城市当前的本地时间并返回 UTC 偏移。用户问“现在几点”“当地时间”时"
         "必须用这个工具，不要自己猜。时差换算请用 calculate。",
         {"type": "object",
          "properties": {"city": {"type": "string", "description": "城市中文名，如 北京"}},
          "required": ["city"]}),
    Tool(calculate,
         "计算一个算术表达式。做时差、数量、百分比等任何数字运算时使用，不要心算。"
         "表达式只能包含数字、括号和 + - * / // % **。",
         {"type": "object",
          "properties": {"expression": {"type": "string", "description": "如 9-8 或 (3+5)*2"}},
          "required": ["expression"]}),
    Tool(read_file,
         "读取工作目录内的一个文本文件。需要查看已有文件内容时使用。",
         {"type": "object",
          "properties": {"path": {"type": "string", "description": "相对路径，如 result.txt"}},
          "required": ["path"]}),
    Tool(write_file,
         "把文本写入工作目录内的文件（会覆盖同名文件）。用户要求保存结论、写报告、"
         "生成文件时使用。",
         {"type": "object",
          "properties": {"path": {"type": "string", "description": "相对路径，如 result.txt"},
                         "content": {"type": "string", "description": "要写入的完整文本"}},
          "required": ["path", "content"]}),
    Tool(list_files,
         "列出工作目录内某个目录的文件和子目录。不知道文件名时先用它。",
         {"type": "object",
          "properties": {"directory": {"type": "string", "description": "相对目录，默认 ."}},
          "required": []}),
]

TOOL_MAP: dict[str, Tool] = {tool.name: tool for tool in TOOLS}

# ---------------------------- Agent ----------------------------
class Agent:
    """持有消息历史、负责循环、负责执行工具。"""

    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self.messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        started = time.perf_counter()

        for turn in range(1, MAX_TURNS + 1):
            elapsed = time.perf_counter() - started
            used = self.prompt_tokens + self.completion_tokens
            if elapsed > MAX_SECONDS:
                return self._abort(f"超过时间预算 {MAX_SECONDS:.0f}s（已用 {elapsed:.1f}s）")
            if used > MAX_TOKENS:
                return self._abort(f"超过 token 预算 {MAX_TOKENS}（已用 {used}）")

            print(f"\n─── 第 {turn} 轮 " + "─" * 42)
            resp = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=[tool.schema() for tool in TOOLS],
                temperature=0.0,
            )
            if resp.usage is not None:
                self.prompt_tokens += resp.usage.prompt_tokens
                self.completion_tokens += resp.usage.completion_tokens

            message = resp.choices[0].message
            tool_calls = [c for c in message.tool_calls or [] if c.type == "function"]

            # 1) assistant 的原始输出必须 append 回历史 —— 包括 tool_calls！
            assistant_msg: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": message.content or "",
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in tool_calls
                ]
            self.messages.append(assistant_msg)

            # 2) 没有工具调用 = 最终回答 = 停止条件之一
            if not tool_calls:
                answer = message.content or "（模型没有返回内容）"
                print(f"[回答] {answer}")
                print(f"[用量] 累计 {self.prompt_tokens + self.completion_tokens} tok | {time.perf_counter() - started:.1f}s")
                return answer

            # 3) 执行工具，把每个结果作为 tool 消息塞回历史
            if message.content:
                print(f"[思考] {message.content.strip()}")
            for call in tool_calls:
                print(f"[调用] {call.function.name}({call.function.arguments})")
                result = self._execute(call.function.name, call.function.arguments)
                print(f"[结果] {result[:160].replace(chr(10), ' ⏎ ')}")
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            print(f"[用量] 累计 {self.prompt_tokens + self.completion_tokens} tok | {time.perf_counter() - started:.1f}s")

        return self._abort(f"达到最大轮次 {MAX_TURNS}")

    def _execute(self, name: str, arguments: str) -> str:
        """执行工具。任何异常都变成字符串返回给模型，绝不让 agent 崩掉。"""
        tool = TOOL_MAP.get(name)
        if tool is None:
            return f"错误：没有名为 {name!r} 的工具。可用：{', '.join(TOOL_MAP)}"
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"错误：参数不是合法 JSON（{exc}）。收到：{arguments!r}"
        if not isinstance(args, dict):
            return "错误：工具参数必须是一个 JSON 对象。"
        try:
            return tool.fn(**args)
        except Exception as exc:
            return f"错误：工具 {name} 执行失败：{type(exc).__name__}: {exc}"

    def _abort(self, reason: str) -> str:
        text = f"[已停止] {reason}。"
        self.messages.append({"role": "assistant", "content": text})
        print(f"\n{text}")
        return text


def make_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("没找到 DEEPSEEK_API_KEY，检查 .env 是否在项目根目录")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def main() -> None:
    agent = Agent(make_client())
    if len(sys.argv) > 1:  # 一次性任务
        agent.run(" ".join(sys.argv[1:]))
        print(f"\n总用量：输入 {agent.prompt_tokens} / 输出 {agent.completion_tokens} token")
        return

    print("交互模式：输入 exit / quit / q 退出。")
    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if user_input:
            agent.run(user_input)


if __name__ == "__main__":
    main()
