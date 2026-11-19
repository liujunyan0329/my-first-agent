"""阶段 2 · 工具注册机制：Pydantic + 装饰器，从函数签名自动生成 JSON Schema。

阶段 1 的 Tool 是手写的：函数、描述、参数 schema 分散在三处，加一个工具要改三遍。
这里用一个 @tool 装饰器把三者绑在一起，并且让 Pydantic：
  1. 从类型注解生成 JSON Schema（发给模型）；
  2. 在执行前校验模型给的参数（挡住类型错误）。

运行演示：uv run python stage2/builtin_tools.py
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, get_type_hints, overload

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, ValidationError, create_model

F = TypeVar("F", bound=Callable[..., Any])


def _clean_schema(node: Any) -> Any:
    """删掉 Pydantic 自动加的 title，减少发给模型的 schema 体积。"""
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    dangerous: bool = False

    def schema(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOLS: dict[str, Tool] = {}


@overload
def tool(fn: F) -> F: ...


@overload
def tool(*, dangerous: bool = False) -> Callable[[F], F]: ...


def tool(
    fn: F | None = None, *, dangerous: bool = False
) -> F | Callable[[F], F]:
    """装饰器：登记一个工具。两种用法：

    @tool
    def add(a: int, b: int) -> int: ...

    @tool(dangerous=True)
    def run_python(code: str) -> str: ...
    """

    def register(func: F) -> F:
        signature = inspect.signature(func)
        hints = get_type_hints(func, include_extras=True)

        fields: dict[str, Any] = {}
        for name, param in signature.parameters.items():
            if name in {"self", "cls"}:
                continue
            annotation = hints.get(name, param.annotation)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (annotation, default)

        # 核心：把函数签名变成一个 Pydantic 模型
        args_model = create_model(f"{func.__name__}Args", **fields)
        raw = args_model.model_json_schema()
        parameters = {
            "type": "object",
            "properties": _clean_schema(raw.get("properties", {})),
            "required": raw.get("required", []),
        }

        doc = inspect.getdoc(func) or func.__name__
        description = " ".join(line.strip() for line in doc.splitlines() if line.strip())

        if func.__name__ in TOOLS:
            raise ValueError(f"工具名重复：{func.__name__}")

        TOOLS[func.__name__] = Tool(
            name=func.__name__,
            description=description,
            parameters=parameters,
            fn=func,
            args_model=args_model,
            dangerous=dangerous,
        )
        return func

    if fn is not None:
        return register(fn)
    return register


def tool_schemas() -> list[ChatCompletionToolParam]:
    """发给模型的工具清单。"""
    return [item.schema() for item in TOOLS.values()]


def execute_tool(name: str, arguments: str | dict[str, Any]) -> str:
    """执行工具：JSON 解析 → Pydantic 校验 → 调用函数 → 异常转成字符串。"""
    item = TOOLS.get(name)
    if item is None:
        return f"错误：没有名为 {name!r} 的工具。可用：{', '.join(TOOLS)}"

    try:
        raw = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError as exc:
        return f"错误：参数不是合法 JSON（{exc}）。收到：{arguments!r}"
    if not isinstance(raw, dict):
        return "错误：工具参数必须是一个 JSON 对象。"

    try:
        args = item.args_model.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        return f"错误：参数校验失败：{details}"

    try:
        return str(item.fn(**args.model_dump()))
    except Exception as exc:
        return f"错误：工具 {name} 执行失败：{type(exc).__name__}: {exc}"
