"""阶段 3 · 实验五：跨会话记忆。两次独立进程，第二次仍然记得第一次的偏好。

运行（必须分两次，才能证明"跨进程"而不是"同一个 Python 变量"）：

    uv run python stage3/memory_demo.py write
    uv run python stage3/memory_demo.py ask "用一句话解释什么是 SAT"

第二次是一个全新的 Python 进程：内存里空无一物，只能从
stage3/runtime/demo_memory.jsonl 里把偏好读回来。

注意：这里用独立的 demo_memory.jsonl，不污染 agent_v3 真正使用的
memory.jsonl。两者结构完全一样，你在交互模式里说"记住……"写的是后者。
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

import memory as memory_module
from memory import MemoryStore

DEMO_PATH = Path(__file__).parent / "runtime" / "demo_memory.jsonl"

SEED = [
    ("称呼我为“舰长”。", "preference"),
    ("回答先给结论，整段不超过 40 个字。", "preference"),
    ("我习惯用 C++17 和 MSYS2，别推荐 Linux 专属方案。", "fact"),
]


def use_demo_store() -> MemoryStore:
    store = MemoryStore(DEMO_PATH)
    memory_module.STORE = store   # 让 agent 的工具和注入都指向同一个 demo 存储
    return store


def write_session() -> None:
    DEMO_PATH.unlink(missing_ok=True)
    store = use_demo_store()
    print(f"写入演示记忆：{DEMO_PATH}\n")
    for text, kind in SEED:
        entry, added = store.add(text, kind=kind)
        print(f"{'新增' if added else '已存在'}  [{entry.kind}] {entry.text}")
    print("\n写入完成。另开一个终端（或现在就）运行：")
    print('  uv run python stage3/memory_demo.py ask "用一句话解释什么是 SAT"')


def ask_session(question: str) -> None:
    store = use_demo_store()
    if not store.entries:
        raise SystemExit(
            "演示记忆是空的。先运行：uv run python stage3/memory_demo.py write"
        )
    print(f"从文件读取到 {len(store.entries)} 条跨会话记忆：")
    for entry in store.entries:
        print(f"  [{entry.kind}] {entry.text}")

    # 延迟导入：只有 ask 才需要 agent 和 API
    from agent_v3 import Agent, make_client

    load_dotenv()
    agent = Agent(make_client(), strategy="summary", budget=6000, use_memory=True)
    print("\n--- 实际注入 system prompt 的记忆块 ---")
    print(store.render() or "（空）")
    print("\n--- 模型回答 ---")
    agent.run(question)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"write", "ask"}:
        print("用法：memory_demo.py write | ask <问题>")
        return
    if sys.argv[1] == "write":
        write_session()
    else:
        if len(sys.argv) < 3:
            raise SystemExit("用法：memory_demo.py ask <问题>")
        ask_session(" ".join(sys.argv[2:]))


if __name__ == "__main__":
    main()
