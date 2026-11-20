"""阶段 3 · 检索层：加载索引、查 top-k、格式化给模型看。

单独抽成一个模块的原因：
  - agent 的工具调用、对比实验、你自己的调试都要用同一套检索逻辑；
  - 检索结果要统一套上 `<UNTRUSTED_DATA>`——RAG 是间接 prompt injection
    最大的入口之一，阶段 2 的教训在这里必须继续用。

命令行调试：
    uv run python stage3/retrieval.py "DPLL 单元传播" --k 5
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

from embedder import embed_query
from rag_store import SearchHit, VectorStore

INDEX_DIR = Path(__file__).parent / "index"


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore.load(INDEX_DIR)


def search_hits(query: str, top_k: int = 4, min_score: float = 0.0) -> list[SearchHit]:
    store = get_store()
    return store.search(
        embed_query(query),
        top_k=top_k,
        min_score=min_score,
        max_per_source=2,   # 同一页最多 2 块，避免 top-k 被一页占满
    )


def search(query: str, top_k: int = 4, min_score: float = 0.0) -> str:
    """检索并格式化。返回值直接作为工具结果交给模型。"""
    hits = search_hits(query, top_k=top_k, min_score=min_score)
    if not hits:
        return "没有检索到相关课程资料。"

    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{index}] {hit.chunk.citation()}（相似度 {hit.score:.3f}）\n"
            f"{hit.chunk.text}"
        )
    body = "\n\n".join(blocks)
    return (
        '<UNTRUSTED_DATA source="课程资料检索结果">\n'
        f"{body}\n"
        "</UNTRUSTED_DATA>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="检索课程资料")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    hits = search_hits(args.query, top_k=args.k, min_score=args.min_score)
    if not hits:
        print("没有结果。")
        return
    for index, hit in enumerate(hits, start=1):
        preview = hit.chunk.text.replace("\n", " ")[:160]
        print(f"{index}. [{hit.score:.3f}] {hit.chunk.citation()}")
        print(f"   {preview}…\n")


if __name__ == "__main__":
    main()
