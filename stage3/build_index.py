"""阶段 3 · 建索引：课程资料 → 切块 → embedding → 落盘。

运行：
    uv run python stage3/build_index.py
    uv run python stage3/build_index.py --size 300 --overlap 50
    uv run python stage3/build_index.py --source "E:/形式化方法" --index stage3/index_v2

第一次运行会自动下载 embedding 模型（约 90MB），国内建议先设置：
    export HF_ENDPOINT=https://hf-mirror.com
建索引是一次性成本：几十秒到几分钟，取决于块数；检索时只加载向量矩阵。
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from chunking import build_chunks
from embedder import DIM, MODEL_NAME, embed_texts
from rag_store import VectorStore

HERE = Path(__file__).parent
DEFAULT_SOURCE = Path("E:/形式化方法")
DEFAULT_INDEX = HERE / "index"


def main() -> None:
    parser = argparse.ArgumentParser(description="为课程资料建向量索引")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--size", type=int, default=500, help="每块目标字数")
    parser.add_argument("--overlap", type=int, default=80, help="相邻块重叠字数")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"资料来源不存在：{args.source}")

    print(f"资料来源：{args.source}")
    started = time.perf_counter()
    chunks = build_chunks(args.source, size=args.size, overlap=args.overlap)
    if not chunks:
        raise SystemExit("没有切出任何块，检查资料来源路径和文件类型")

    source_counts = Counter(chunk.source for chunk in chunks)
    total_chars = sum(len(chunk.text) for chunk in chunks)
    print(f"切块完成：{len(chunks)} 块，{total_chars} 字，"
          f"{len(source_counts)} 个来源，用时 {time.perf_counter() - started:.1f}s")
    for source, count in source_counts.most_common():
        print(f"  {count:>4} 块  {source}")

    started = time.perf_counter()
    vectors = embed_texts([chunk.text for chunk in chunks])
    elapsed = time.perf_counter() - started
    print(f"\nembedding 完成：{vectors.shape}，{elapsed:.1f}s "
          f"（{len(chunks) / max(elapsed, 1e-6):.1f} 块/秒）")

    store = VectorStore(
        chunks=chunks,
        vectors=vectors,
        meta={
            "model": MODEL_NAME,
            "dim": DIM,
            "chunk_size": args.size,
            "overlap": args.overlap,
            "source_root": str(args.source),
            "sources": dict(source_counts),
        },
    )
    store.save(args.index)
    print(f"索引已保存：{args.index}（vectors.npy / chunks.jsonl / meta.json）")


if __name__ == "__main__":
    main()
