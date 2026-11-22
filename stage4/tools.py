"""阶段 4 · 复用的工具集。

直接复用阶段 2 的 @tool 注册机制（工具函数 + 类型注解 + docstring → JSON Schema），
只是把工作目录换到 stage4/workspace，并新增一个故意带 I/O 延迟的 mock 工具，
用来做"串行 vs 并行"的延迟对比。
"""

from __future__ import annotations

import ast
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from pydantic import Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "stage2"))

from registry import TOOLS, execute_tool, tool, tool_schemas  # noqa: E402

WORKSPACE = ROOT / "workspace"
WORKSPACE.mkdir(exist_ok=True)

CITY_OFFSETS: dict[str, int] = {
    "北京": 8, "上海": 8, "东京": 9, "首尔": 9, "悉尼": 11,
    "伦敦": 0, "巴黎": 1, "柏林": 1, "纽约": -5, "洛杉矶": -8,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)


def _resolve(path: str) -> Path:
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"路径越界：{path!r} 不在工作目录 {WORKSPACE.name} 内")
    return target


@tool
def get_current_time(
    city: Annotated[str, Field(description="城市中文名，如 北京、东京")],
) -> str:
    """查询一个城市当前的本地时间并返回 UTC 偏移。用户问“现在几点”时使用，不要自己猜。"""
    offset = CITY_OFFSETS.get(city.strip())
    if offset is None:
        return f"错误：不认识城市 {city!r}。支持：{', '.join(CITY_OFFSETS)}"
    now = datetime.now(timezone(timedelta(hours=offset)))
    return f"{city}现在是 {now:%Y-%m-%d %H:%M:%S}（UTC{offset:+d}）"


@tool
def calculate(
    expression: Annotated[
        str, Field(description="算术表达式，只能包含数字、括号和 + - * / // % **")
    ],
) -> str:
    """计算一个算术表达式。做时差、数量、百分比等数字运算时使用，不要心算。"""
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
        return "错误：除数不能为 0。"
    except Exception as exc:
        return f"错误：计算失败：{type(exc).__name__}: {exc}"
    return f"{expression} = {value}"


@tool
def read_file(
    path: Annotated[str, Field(description="相对 workspace 的路径，如 result.txt")],
) -> str:
    """读取 workspace 内的一个文本文件。需要查看已有文件内容时使用。"""
    target = _resolve(path)
    if not target.is_file():
        return f"错误：文件不存在：{path!r}。可以先用 list_files 看看有什么。"
    text = target.read_text(encoding="utf-8")
    if len(text) > 4000:
        return text[:4000] + f"\n…（已截断，原文共 {len(text)} 字符）"
    return text


@tool(dangerous=True)
def write_file(
    path: Annotated[str, Field(description="相对 workspace 的路径，如 result.txt")],
    content: Annotated[str, Field(description="要写入的完整文本")],
) -> str:
    """把文本写入 workspace 内的文件（会覆盖同名文件）。用户要求保存结论时使用。这是危险工具。"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {target.relative_to(WORKSPACE)}（{len(content)} 字符）"


@tool
def list_files(
    directory: Annotated[str, Field(description="相对 workspace 的目录，默认 .")] = ".",
) -> str:
    """列出 workspace 内某个目录的文件和子目录。不知道文件名时先用它。"""
    target = _resolve(directory)
    if not target.is_dir():
        return f"错误：目录不存在：{directory!r}"
    entries = sorted(target.iterdir())[:50]
    lines = [("目录 " if p.is_dir() else "文件 ") + p.name for p in entries]
    return "\n".join(lines) if lines else "（空目录）"


# ---- 只用于"并行 vs 串行"实验：模拟一个要走网络的慢接口 ----
MOCK_WEATHER: dict[str, str] = {
    "北京": "晴，18℃，西北风 2 级",
    "上海": "多云，21℃，东风 1 级",
    "东京": "小雨，16℃，南风 3 级",
    "纽约": "阴，12℃，西风 4 级",
}
MOCK_LATENCY = 0.8  # 秒，模拟一次 HTTP 往返


@tool
def mock_weather(
    city: Annotated[str, Field(description="城市中文名，如 北京")],
) -> str:
    """查询城市天气（模拟接口，每次固定耗时 0.8 秒）。做并行实验时使用。"""
    time.sleep(MOCK_LATENCY)
    weather = MOCK_WEATHER.get(city.strip())
    if weather is None:
        return f"错误：没有 {city!r} 的天气数据。支持：{', '.join(MOCK_WEATHER)}"
    return f"{city}：{weather}"


__all__ = [
    "TOOLS", "WORKSPACE", "calculate", "execute_tool", "get_current_time",
    "list_files", "mock_weather", "read_file", "tool_schemas", "write_file",
]
