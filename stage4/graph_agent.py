"""阶段 4 · 用 LangGraph 重写阶段 1 的 agent 循环。

阶段 1 的循环是手写的：
    while turn < MAX_TURNS:
        resp = model(messages)
        if no tool_calls: return answer
        for call in tool_calls: messages.append(execute(call))

这里把同一个循环拆成"图"：
    节点 = 一次计算（调模型 / 跑工具 / 中止）
    边   = 流转（模型之后走工具还是结束）
    状态 = 共享内存（messages、turn、token 计数）

运行：
    uv run python stage4/graph_agent.py "北京现在几点？比东京早还是晚几小时？"
    uv run python stage4/graph_agent.py "把结论写进 stage4-result.txt"
    uv run python stage4/graph_agent.py --parallel "同时查北京、东京、纽约的时间"
    uv run python stage4/graph_agent.py --graph           # 只打印 Mermaid 图
"""

from __future__ import annotations

import argparse
import asyncio
import operator
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict, cast

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools import execute_tool, tool_schemas  # noqa: E402

load_dotenv()

MODEL = "deepseek-chat"
MAX_TURNS = 8
PRICE_INPUT = 2.0 / 1_000_000
PRICE_OUTPUT = 8.0 / 1_000_000

SYSTEM_PROMPT = """\
你是一个会使用工具的助手。规则：
1. 不知道的事实必须查：时间用 get_current_time，算术用 calculate，
   文件内容用 read_file / list_files，保存结论用 write_file。不要凭空编造。
2. 相互独立的工具调用可以放在同一条消息里一起发起。
3. 工具返回以“错误：”开头时，先读错误信息，修正参数后重试。
4. 完成用户要求后，用简洁的中文给出最终结论，不要再调用工具。"""

Message = dict[str, Any]


class AgentState(TypedDict):
    """图里流转的共享状态。

    messages 用 operator.add 作为 reducer：节点只返回"新增的消息"，
    LangGraph 负责把它们追加到列表末尾。这是图式编排和手写循环最大的写法差异。
    """

    messages: Annotated[list[Message], operator.add]
    turns: int
    prompt_tokens: int
    completion_tokens: int
    answer: str
    abort_reason: str


def _assistant_message(message: Any) -> Message:
    tool_calls = [c for c in message.tool_calls or [] if c.type == "function"]
    assistant: Message = {"role": "assistant", "content": message.content or ""}
    if tool_calls:
        assistant["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return assistant


def build_graph(client: OpenAI, *, parallel: bool = False):
    """构造图。返回编译后的 graph（还没绑定 checkpointer）。"""

    def call_model(state: AgentState) -> dict[str, Any]:
        turn = state["turns"] + 1
        print(f"\n─── 第 {turn} 轮 · 节点 model " + "─" * 36)
        response = client.chat.completions.create(
            model=MODEL,
            messages=state["messages"],  # type: ignore[arg-type]
            tools=tool_schemas(),
            temperature=0.0,
        )
        message = response.choices[0].message
        assistant = _assistant_message(message)
        usage = response.usage
        prompt_tokens = state["prompt_tokens"] + (usage.prompt_tokens if usage else 0)
        completion_tokens = state["completion_tokens"] + (
            usage.completion_tokens if usage else 0
        )

        if message.content and assistant.get("tool_calls"):
            print(f"[思考] {message.content.strip()[:300]}")
        if not assistant.get("tool_calls"):
            print(f"[回答] {assistant['content']}")
        return {
            "messages": [assistant],
            "turns": turn,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "answer": str(assistant["content"]) if not assistant.get("tool_calls") else "",
        }

    def run_one(call: Message) -> Message:
        function = call["function"]
        name = function["name"]
        arguments = function["arguments"]
        print(f"[调用] {name}({arguments})")
        result = execute_tool(name, arguments)
        print(f"[结果] {result[:200].replace(chr(10), ' ⏎ ')}")
        return {"role": "tool", "tool_call_id": call["id"], "content": result}

    async def run_parallel(calls: list[Message]) -> list[Message]:
        # execute_tool 是同步函数，用 to_thread 把它丢到线程池，事件循环才不会被它卡住。
        return list(
            await asyncio.gather(*(asyncio.to_thread(run_one, call) for call in calls))
        )

    def call_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(last.get("tool_calls") or [])
        print(f"─── 节点 tools：{len(calls)} 个调用 " + "─" * 36)
        if parallel and len(calls) > 1:
            started = time.perf_counter()
            results = asyncio.run(run_parallel(calls))
            print(f"[并行] {len(calls)} 个工具耗时 {time.perf_counter() - started:.2f}s")
        else:
            results = [run_one(call) for call in calls]
        return {"messages": results}

    def abort(state: AgentState) -> dict[str, Any]:
        reason = f"达到最大轮次 {MAX_TURNS}，任务未完成。"
        print(f"\n[中止] {reason}")
        return {
            "messages": [{"role": "assistant", "content": reason}],
            "answer": reason,
            "abort_reason": reason,
        }

    def after_model(state: AgentState) -> str:
        last = state["messages"][-1]
        if not last.get("tool_calls"):
            return END
        if state["turns"] >= MAX_TURNS:
            return "abort"
        return "tools"

    builder = StateGraph(AgentState)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_node("abort", abort)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", after_model, {"tools": "tools", "abort": "abort", END: END})
    builder.add_edge("tools", "model")
    builder.add_edge("abort", END)
    return builder.compile()


def make_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填写")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def run(client: OpenAI, task: str, *, parallel: bool) -> tuple[str, AgentState, float]:
    graph = build_graph(client, parallel=parallel)
    started = time.perf_counter()
    state = cast(AgentState, graph.invoke(
        {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": task}],
         "turns": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "answer": "", "abort_reason": ""},
        {"recursion_limit": 2 * MAX_TURNS + 4},
    ))
    elapsed = time.perf_counter() - started
    used = state["prompt_tokens"] + state["completion_tokens"]
    cost = state["prompt_tokens"] * PRICE_INPUT + state["completion_tokens"] * PRICE_OUTPUT
    print(
        f"\n[用量] {used} tok（输入 {state['prompt_tokens']} / 输出 "
        f"{state['completion_tokens']}），约 ¥{cost:.4f}，{elapsed:.1f}s，"
        f"共 {state['turns']} 轮"
    )
    return state["answer"], state, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 4 · LangGraph 版 agent")
    parser.add_argument("task", nargs="*", help="一次性任务；省略则进入交互模式")
    parser.add_argument("--parallel", action="store_true", help="工具调用并行执行")
    parser.add_argument("--graph", action="store_true", help="打印 Mermaid 图后退出")
    args = parser.parse_args()

    client = make_client()
    if args.graph:
        print(build_graph(client).get_graph().draw_mermaid())
        return

    if args.task:
        run(client, " ".join(args.task), parallel=args.parallel)
        return

    print("交互模式：输入 exit / quit / q 退出。")
    while True:
        try:
            text = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"exit", "quit", "q"}:
            break
        if text:
            run(client, text, parallel=args.parallel)


if __name__ == "__main__":
    main()
