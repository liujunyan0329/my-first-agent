"""阶段 4 · human-in-the-loop：危险动作前暂停，等人批准后再继续。

流程：
    draft（模型起草内容） → approve（interrupt 暂停） → execute（写文件） / cancelled

关键点：
  1. interrupt() 不是 input()：它把整个图的状态存进检查点，进程可以退出；
  2. 批准/拒绝发生在另一个进程里，用 Command(resume=...) 把决定送回图；
  3. 暂停期间没有任何线程在等——服务器重启、用户第二天再来都可以。

运行：
    uv run python stage4/hitl_demo.py run "写一份本周学习计划"
    uv run python stage4/hitl_demo.py approve
    uv run python stage4/hitl_demo.py reject
    uv run python stage4/hitl_demo.py status
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
DB_PATH = RUNTIME / "hitl.sqlite"
WORKSPACE = ROOT / "workspace"
THREAD_ID = "stage4-hitl-demo"
MODEL = "deepseek-chat"

load_dotenv()


class HitlState(TypedDict):
    task: str
    draft: str
    path: str
    decision: str
    result: str


def make_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def build_graph(client: OpenAI, checkpointer: SqliteSaver):
    def draft(state: HitlState) -> dict[str, Any]:
        print("[节点 draft] 让模型起草内容", flush=True)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是文本助手。只输出正文，不要解释，不超过 120 字。"},
                {"role": "user", "content": f"请为下面这件事写一段备忘录正文：{state['task']}"},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        content = (response.choices[0].message.content or "").strip()
        path = "hitl-note.md"
        print(f"[节点 draft] 起草完成，{len(content)} 字", flush=True)
        return {"draft": content, "path": path}

    def approve(state: HitlState) -> dict[str, Any]:
        # interrupt 把 payload 交给调用方；函数在这里"冻结"。
        # 恢复时，interrupt() 的返回值就是 Command(resume=...) 带回来的值。
        decision = interrupt(
            {
                "action": "write_file",
                "path": state["path"],
                "preview": state["draft"][:200],
                "question": "允许写入这个文件吗？回答 approve 或 reject。",
            }
        )
        print(f"[节点 approve] 收到人工决定：{decision}", flush=True)
        return {"decision": str(decision)}

    def execute(state: HitlState) -> dict[str, Any]:
        target = WORKSPACE / state["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(state["draft"] + "\n", encoding="utf-8")
        print(f"[节点 execute] 已写入 {target}", flush=True)
        return {"result": f"已写入 {state['path']}（{len(state['draft'])} 字）"}

    def cancelled(state: HitlState) -> dict[str, Any]:
        print("[节点 cancelled] 人工拒绝，没有写任何文件", flush=True)
        return {"result": "已取消，未写入文件"}

    def after_approve(state: HitlState) -> str:
        return "execute" if state["decision"] == "approve" else "cancelled"

    builder = StateGraph(HitlState)
    builder.add_node("draft", draft)
    builder.add_node("approve", approve)
    builder.add_node("execute", execute)
    builder.add_node("cancelled", cancelled)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "approve")
    builder.add_conditional_edges(
        "approve", after_approve, {"execute": "execute", "cancelled": "cancelled"}
    )
    builder.add_edge("execute", END)
    builder.add_edge("cancelled", END)
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 4 · human-in-the-loop 演示")
    parser.add_argument("action", choices=["run", "approve", "reject", "status"])
    parser.add_argument("task", nargs="*", help="run 时的任务描述")
    parser.add_argument("--thread", default=THREAD_ID)
    args = parser.parse_args()

    RUNTIME.mkdir(parents=True, exist_ok=True)
    config: RunnableConfig = {"configurable": {"thread_id": args.thread}}
    client = make_client()

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(client, checkpointer)

        if args.action == "run":
            task = " ".join(args.task) or "写一份本周学习计划"
            initial: HitlState = {
                "task": task, "draft": "", "path": "", "decision": "", "result": "",
            }
            result = graph.invoke(initial, config)
            interrupts = result.get("__interrupt__") or []
            if interrupts:
                payload = interrupts[0].value
                print("\n⏸  图已暂停，等待人工批准：")
                print(f"    动作：{payload['action']}  {payload['path']}")
                print(f"    内容预览：{payload['preview']}")
                print(f"\n    批准：uv run python stage4/hitl_demo.py approve --thread {args.thread}")
                print(f"    拒绝：uv run python stage4/hitl_demo.py reject  --thread {args.thread}")
            else:
                print(f"没有中断，结果：{result.get('result')}")
            return

        if args.action == "status":
            snapshot = graph.get_state(config)
            print(f"已完成值：{snapshot.values}")
            print(f"待执行节点：{snapshot.next}")
            return

        decision = "approve" if args.action == "approve" else "reject"
        print(f"把决定 '{decision}' 送回图（新的进程也可以）", flush=True)
        result = graph.invoke(Command(resume=decision), config)
        print(f"\n最终结果：{result.get('result')}")
        if args.action == "approve":
            target = WORKSPACE / str(result.get("path", ""))
            print(f"文件内容：\n{target.read_text(encoding='utf-8') if target.exists() else '（不存在）'}")


if __name__ == "__main__":
    main()
