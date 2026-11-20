"""阶段 3 · 最小向量库：numpy 矩阵 + 余弦相似度 + 落盘。

真实项目用 Chroma / FAISS / pgvector，但它们的核心动作就这三步：
  1. 把每块文档的向量存成一个矩阵 (n_chunks, dim)；
  2. 查询向量和矩阵做一次矩阵乘法；
  3. 排序取 top-k。

数据量在几千块以内时，暴力搜索比"建索引"还快，别急着上数据库。

索引目录结构：
    index/
    ├── vectors.npy     # (n, 512) float32，已归一化
    ├── chunks.jsonl    # 每行一块：chunk_id / source / page / section / text
    └── meta.json       # 模型名、维度、切块参数、构建时间、来源统计
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from chunking import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"块数 {len(chunks)} 与向量数 {len(vectors)} 不一致，索引损坏"
            )
        self.chunks = chunks
        self.vectors = vectors
        self.meta = meta or {}

    # ------------------------------ 持久化 ------------------------------
    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as file:
            for chunk in self.chunks:
                record = {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "section": chunk.section,
                    "text": chunk.text,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta = {
            **self.meta,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chunk_count": len(self.chunks),
            "dim": int(self.vectors.shape[1]),
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path | str) -> VectorStore:
        directory = Path(directory)
        vectors_path = directory / "vectors.npy"
        chunks_path = directory / "chunks.jsonl"
        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"索引不存在：{directory}\n先运行 uv run python stage3/build_index.py"
            )
        vectors = np.load(vectors_path)
        chunks: list[Chunk] = []
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=record["chunk_id"],
                    source=record["source"],
                    page=record["page"],
                    section=record["section"],
                    text=record["text"],
                )
            )
        meta_path = directory / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(chunks, vectors, meta)

    # ------------------------------ 检索 ------------------------------
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        min_score: float = 0.0,
        max_per_source: int | None = None,
    ) -> list[SearchHit]:
        """余弦相似度检索。

        向量都已归一化，所以 `vectors @ query` 直接就是余弦相似度。
        max_per_source 用来避免 top-k 全被同一页/同一份文档占满。
        """
        if self.vectors.shape[0] == 0:
            return []
        scores = self.vectors @ query_vector
        order = np.argsort(-scores)

        hits: list[SearchHit] = []
        per_source: dict[str, int] = {}
        for index in order:
            score = float(scores[index])
            if score < min_score:
                break
            chunk = self.chunks[int(index)]
            if max_per_source is not None:
                key = chunk.citation()
                if per_source.get(key, 0) >= max_per_source:
                    continue
                per_source[key] = per_source.get(key, 0) + 1
            hits.append(SearchHit(chunk, score))
            if len(hits) >= top_k:
                break
        return hits
