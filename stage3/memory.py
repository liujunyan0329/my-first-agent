"""阶段 3 · 跨会话长期记忆：把用户偏好/事实存到本地文件，重启后仍然记得。

短期记忆（context 管理）只活在一个会话里；进程一退就没了。
长期记忆的关键是**持久化 + 检索**：

    会话 A: remember("用户喜欢简洁回答") ──→ memory.jsonl
    会话 B: 启动时读取 → 按当前问题检索 → 塞进 system prompt → 模型表现不同

这里刻意先用"文件 + 关键词检索"实现，让你看清记忆系统的三个基本动作：
写入（去重）、检索（排序）、注入（有预算上限）。
语义检索用 `recall_semantic()` 接到阶段 3 的 embedding 上，作为练习。

安全提示：记忆是持久化的间接注入入口——不可信内容一旦写进记忆，
以后每个会话都会读到它。写入前要确认来源，并限制条数。

运行演示见：uv run python stage3/memory_demo.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_PATH = Path(__file__).parent / "runtime" / "memory.jsonl"
MAX_ENTRIES = 50          # 记忆不是聊天记录，必须是有界的
MAX_RENDER_CHARS = 1200   # 注入 system prompt 的上限


class MemoryEntry(BaseModel):
    text: str
    kind: str = "fact"              # fact / preference / decision
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    source: str = "user"            # 谁写的：user / tool / imported


def _normalize(text: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:]", "", text).lower()


class MemoryStore:
    """JSONL 追加写，全量读入内存。50 条以内，不需要数据库。"""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.entries: list[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                self.entries.append(MemoryEntry.model_validate_json(line))
            except ValueError:
                # 坏行跳过，不要让一条脏数据毁掉全部记忆
                print(f"[memory] 跳过第 {line_number} 行：不是合法的记忆条目")

    def _append(self, entry: MemoryEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(entry.model_dump_json() + "\n")

    def add(
        self,
        text: str,
        kind: str = "fact",
        source: str = "user",
    ) -> tuple[MemoryEntry, bool]:
        """写入一条记忆。返回 (条目, 是否新增)。

        去重规则：归一化后完全相同 = 同一条，不重复写。
        冲突（"喜欢长回答" vs "喜欢短回答"）这里不做自动覆盖——
        真实系统需要显式 forget/update，否则模型会把两条矛盾记忆都读进去。
        """
        text = " ".join(text.split())
        if not text:
            raise ValueError("不能写入空记忆")
        key = _normalize(text)
        for entry in self.entries:
            if _normalize(entry.text) == key:
                return entry, False

        entry = MemoryEntry(text=text, kind=kind, source=source)
        self.entries.append(entry)
        self._append(entry)

        # 超出上限时，从最旧的开始丢（简单 LRU 思路的第一步）
        if len(self.entries) > MAX_ENTRIES:
            self.entries = self.entries[-MAX_ENTRIES:]
            self._rewrite()
        return entry, True

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(entry.model_dump_json() + "\n" for entry in self.entries),
            encoding="utf-8",
        )

    def all(self) -> list[MemoryEntry]:
        return list(self.entries)

    def search(self, query: str, top_k: int = 3) -> list[MemoryEntry]:
        """关键词检索：中文按字二元组、英文按单词，重叠越多分越高。

        这不是语义检索（"喜欢简洁" 搜不到 "回答要短"），
        但足够展示记忆层的工作方式；语义版见 recall_semantic() 的注释。
        """
        terms = _terms(query)
        if not terms:
            return self.entries[-top_k:]
        scored: list[tuple[float, MemoryEntry]] = []
        for index, entry in enumerate(self.entries):
            overlap = len(terms & _terms(entry.text))
            if overlap:
                # 轻微偏向新记忆，避免旧偏好永远压过新偏好
                scored.append((overlap + index / (len(self.entries) + 1), entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def render(self, query: str | None = None, limit: int = MAX_RENDER_CHARS) -> str:
        """生成注入 system prompt 的文本块，按字符数截断。"""
        entries = self.search(query, top_k=5) if query else []
        if not entries:
            # 关键词没命中时退回最近的记忆：宁可在预算内多带几条，
            # 也不要因为一个查询词没对上就让模型"失忆"。
            entries = self.entries[-5:]
        if not entries:
            return ""
        lines = ["## 长期记忆（跨会话，来自之前的对话）"]
        for entry in entries:
            lines.append(f"- [{entry.kind}] {entry.text}")
        text = "\n".join(lines)
        if len(text) > limit:
            text = text[:limit] + "…（已截断）"
        return text


def _terms(text: str) -> set[str]:
    """中文取二元组，英文取小写单词。"""
    lowered = text.lower()
    ascii_words = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    cjk = "".join(re.findall(r"[一-鿿]", lowered))
    bigrams = {cjk[i:i + 2] for i in range(len(cjk) - 1)}
    return ascii_words | bigrams


STORE = MemoryStore()


def remember(text: str, kind: str = "fact") -> str:
    """存一条关于用户的长期记忆（偏好、事实、决定），跨会话有效。

    只在用户明确表达了稳定偏好或重要事实时使用；不要保存密码、密钥、
    身份证号等敏感信息，也不要把工具/网页返回的内容原样写进来。
    """
    entry, added = STORE.add(text, kind=kind)
    if not added:
        return f"记忆已存在，未重复写入：{entry.text}"
    return f"已记住（{entry.kind}）：{entry.text}"


def recall(query: str, top_k: int = 3) -> str:
    """按关键词检索长期记忆。需要回忆用户之前的偏好或事实时使用。"""
    entries = STORE.search(query, top_k=top_k)
    if not entries:
        return "没有找到相关记忆。"
    return "\n".join(
        f"- [{entry.kind}] {entry.text}（{entry.created_at[:10]}）"
        for entry in entries
    )


# 语义检索的扩展点（练习）：
#   from embedder import embed_texts
#   vecs = embed_texts([e.text for e in self.entries] + [query])
#   scores = vecs[:-1] @ vecs[-1]
# 记忆超过几百条、关键词召回明显不够时再换，别提前复杂化。


def _main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        for entry in STORE.all():
            print(f"{entry.created_at}  [{entry.kind}]  {entry.text}")
        if not STORE.entries:
            print("（记忆为空）")
        return
    if len(sys.argv) > 2 and sys.argv[1] == "add":
        print(remember(sys.argv[2], kind=sys.argv[3] if len(sys.argv) > 3 else "fact"))
        return
    if len(sys.argv) > 2 and sys.argv[1] == "search":
        print(recall(sys.argv[2]))
        return
    print("用法：memory.py list | add <文本> [kind] | search <查询>")


if __name__ == "__main__":
    _main()
