"""阶段 4 · 持久化执行（durable execution）：进程被杀后从检查点继续。

流水线：collect → analyze → barrier → crash_point → report。
第一次运行在 crash_point 节点里主动 os._exit(7)，模拟断电/崩溃。
crash_point 能开始执行，说明它前面的 analyze 检查点已经提交。
第二次运行用同一个 thread_id 调 graph.invoke(None, config)，
LangGraph 会从 SQLite 里的最后一个检查点恢复，**不会重跑 collect / analyze**。

运行：
    uv run python stage4/checkpoint_demo.py --crash    # 第一次：跑一半被杀
    uv run python stage4/checkpoint_demo.py --resume   # 第二次：从断点继续
    uv run python stage4/checkpoint_demo.py --history  # 查看检查点历史
"""

from __future__ import annotations

import argparse
import operator
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
DB_PATH = RUNTIME / "checkpoints.sqlite"
LOG_PATH = RUNTIME / "checkpoint_demo.log"
REPORT_PATH = ROOT / "workspace" / "checkpoint-report.md"
THREAD_ID = "stage4-durable-demo"


class DemoState(TypedDict):
    task: str
    log: Annotated[list[str], operator.add]
    plan: str
    draft: str
    report_path: str


def side_effect(text: str) -> None:
    """把"这个节点真的执行过"写到磁盘。进程被杀后，文件就是证据。"""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(f"{stamp}  {text}\n")


def collect(state: DemoState) -> dict[str, Any]:
    print("\n[节点 collect] 收集资料（模拟 1 秒 I/O）", flush=True)
    time.sleep(1.0)
    side_effect("collect 执行了")
    return {
        "log": ["collect"],
        "plan": "1. 收集资料 → 2. 分析要点 → 3. 生成报告",
    }


def analyze(state: DemoState) -> dict[str, Any]:
    print("[节点 analyze] 分析资料（模拟 1 秒 I/O）", flush=True)
    time.sleep(1.0)
    side_effect("analyze 执行了")
    return {
        "log": ["analyze"],
        "draft": (
            "# 阶段 4 检查点演示报告\n\n"
            "- collect：已完成\n"
            "- analyze：已完成\n"
            "- report：由断点恢复后生成\n"
        ),
    }


def barrier(state: DemoState) -> dict[str, Any]:
    """一个什么都不做的节点。

    它的作用是确认上一个节点（analyze）的检查点已经提交：
    图能走到 barrier，说明 analyze 的检查点已经落盘。
    """
    print("[节点 barrier] 空节点，用于确认上一步已提交", flush=True)
    return {}


def crash_point(state: DemoState) -> dict[str, Any]:
    """崩溃点。

    图能开始执行这个节点，说明 barrier 和 analyze 的检查点都已经提交。
    在这里 os._exit，恢复时只会重跑 crash_point 和 report，不会重跑 analyze。
    """
    print("[节点 crash_point] 检查是否要模拟断电", flush=True)
    if os.environ.get("STAGE4_CRASH") == "1":
        print("  💥 模拟断电：os._exit(7)，不做任何清理", flush=True)
        os._exit(7)
    return {}


def report(state: DemoState) -> dict[str, Any]:
    print("[节点 report] 写报告", flush=True)
    side_effect("report 开始")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(state["draft"], encoding="utf-8")
    side_effect("report 完成")
    return {"log": ["report"], "report_path": str(REPORT_PATH)}


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(DemoState)
    builder.add_node("collect", collect)
    builder.add_node("analyze", analyze)
    builder.add_node("barrier", barrier)
    builder.add_node("crash_point", crash_point)
    builder.add_node("report", report)
    builder.add_edge(START, "collect")
    builder.add_edge("collect", "analyze")
    builder.add_edge("analyze", "barrier")
    builder.add_edge("barrier", "crash_point")
    builder.add_edge("crash_point", "report")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)


def reset() -> None:
    for path in (DB_PATH, LOG_PATH, REPORT_PATH):
        if path.exists():
            path.unlink()
    print("已清理旧的检查点、日志和报告。")


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 4 · 断点恢复演示")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--crash", action="store_true", help="第一次运行：跑到 crash_point 时崩溃")
    group.add_argument("--resume", action="store_true", help="第二次运行：从检查点继续")
    group.add_argument("--history", action="store_true", help="查看检查点历史")
    args = parser.parse_args()

    RUNTIME.mkdir(parents=True, exist_ok=True)
    config: RunnableConfig = {"configurable": {"thread_id": THREAD_ID}}

    # Windows 上 SQLite 文件被连接占用时不能删除，所以先清理再打开连接。
    if args.crash:
        reset()

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(checkpointer)

        if args.crash:
            print(f"第一次运行（thread_id={THREAD_ID}）", flush=True)
            initial: DemoState = {
                "task": "写一份阶段 4 检查点演示报告",
                "log": [], "plan": "", "draft": "", "report_path": "",
            }
            os.environ["STAGE4_CRASH"] = "1"
            for update in graph.stream(initial, config, stream_mode="updates", durability="sync"):
                node = next(iter(update))
                print(f"[检查点已保存] 节点 {node} 完成", flush=True)
            # 正常不会到这里：crash_point 里已经 os._exit(7)
            print("没有触发崩溃，检查 STAGE4_CRASH。", flush=True)
            return

        if args.resume:
            os.environ.pop("STAGE4_CRASH", None)
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise SystemExit("没有找到检查点，请先运行 --crash。")
            print(f"恢复前检查点：已完成 {snapshot.values['log']}")
            print("（崩溃时正在执行的节点可能重跑一次——断点粒度是节点，不是语句）")
            print("\n第二次运行：graph.invoke(None, config)")
            final = graph.invoke(None, config, durability="sync")
            print(f"\n最终状态：{final['log']} → {final['report_path']}")
            print(f"\n磁盘上的执行日志：{LOG_PATH}")
            print(LOG_PATH.read_text(encoding="utf-8"))
            return

        # --history
        print(f"检查点历史（thread_id={THREAD_ID}）：")
        for index, snapshot in enumerate(graph.get_state_history(config), start=1):
            values = snapshot.values or {}
            print(
                f"  {index}. next={snapshot.next} "
                f"已完成={values.get('log', [])} "
                f"checkpoint_id={str((snapshot.config.get('configurable') or {}).get('checkpoint_id', '?'))[:8]}"
            )


if __name__ == "__main__":
    main()
