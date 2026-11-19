"""阶段 2 · 用 @tool 注册的 6 个工具。

看这个文件时重点注意两件事：
  1. 每个工具只写函数 + 类型注解 + docstring，不再手写 JSON Schema；
  2. 参数说明写在 Annotated[..., Field(description="...")] 里，
     Pydantic 会把它变成 JSON Schema 的 description。

运行演示：uv run python stage2/builtin_tools.py
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from pydantic import Field

from registry import TOOLS, execute_tool, tool
from sandbox import run_code

WORKSPACE = Path(__file__).resolve().parent / "workspace"
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


@tool
def get_current_time(
    city: Annotated[str, Field(description="城市中文名，如 北京、东京")],
) -> str:
    """查询一个城市当前的本地时间并返回 UTC 偏移。用户问“现在几点”“当地时间”时使用，不要自己猜。"""
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


def _resolve(path: str) -> Path:
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"路径越界：{path!r} 不在工作目录 {WORKSPACE.name} 内")
    return target


@tool
def read_file(
    path: Annotated[str, Field(description="相对 workspace 的路径，如 notes/meeting.md")],
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


@tool(dangerous=True)
def run_python(
    code: Annotated[
        str, Field(description="要执行的 Python 代码。只写计算/数据处理，不要做危险操作。")
    ],
) -> str:
    """在沙箱中执行一段 Python 代码，返回 stdout/stderr。需要真正运行代码来计算或处理数据时使用。这是危险工具。"""
    return run_code(code, timeout=5.0).as_text()


if __name__ == "__main__":
    print("=" * 60)
    print("注册的工具 schema（只截取前两个）")
    for item in list(TOOLS.values())[:2]:
        print(json.dumps(item.schema(), ensure_ascii=False, indent=2))

    print("=" * 60)
    print("Pydantic 校验演示")
    print("正常：", execute_tool("calculate", {"expression": "9-8"}))
    print("类型错：", execute_tool("calculate", {"expression": 123}))
    print("缺参数：", execute_tool("get_current_time", {}))
    print("未知工具：", execute_tool("no_such_tool", {}))
    print("=" * 60)
    print("沙箱演示：", execute_tool("run_python", {"code": "print(sum(range(1, 101)))"}))
