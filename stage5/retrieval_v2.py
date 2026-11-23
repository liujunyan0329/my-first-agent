"""阶段 5 · 升级版检索：标题注入 + BM25/向量混合（RRF）。

阶段 3 的最小 RAG 只做了"向量 top-k"。在 eval 里你会看到它的两个典型失败：

  1. **标题不在块里**。Markdown 切块时标题被单独存进 `section`，正文里没有
     "SAT 与 SMT" 这几个字，于是问"SMT 和 SAT 的区别"时检索不到定义页。
     解决办法：建索引时把 `来源 · 章节` 拼在正文前面再 embed（标题注入）。

  2. **专有名词被语义相似度稀释**。问 "CDCL 相比 DPLL 的改进"，向量可能把
     "DPLL 与 CDCL" 这一节排在后面。解决办法：加一路 BM25 关键词检索，
     再把两路分数归一化后融合：

         score = 0.5 * minmax(cosine) + 0.5 * minmax(BM25)

     也试过 RRF（只按排名融合），但它在"BM25 第 1、向量第 200"这种块上会
     输给"两边都是第 10"的块，专有名词查询反而被压下去。分数融合更稳。

本模块同时保留"纯向量"模式，用来做对照实验。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "stage3"))

from embedder import embed_query  # noqa: E402
from rag_store import SearchHit, VectorStore  # noqa: E402

Mode = Literal["vector", "hybrid"]


def tokenize(text: str) -> list[str]:
    """中文用字符二元组，英文用单词。够用，且没有额外依赖。

    中文没有空格，按"字二元组"切是最便宜的分词近似：
    "单元传播" -> ["单元", "元传", "传播"]。
    """
    lowered = text.lower()
    latin = re.findall(r"[a-z][a-z0-9_+#.-]+", lowered)
    han = re.findall(r"[一-鿿]", lowered)
    bigrams = ["".join(han[i : i + 2]) for i in range(len(han) - 1)]
    return latin + bigrams


def embedding_text(chunk: object) -> str:
    """建索引时真正送去 embed 的文本：标题/来源 + 正文。"""
    source = getattr(chunk, "source", "")
    section = getattr(chunk, "section", "")
    text = getattr(chunk, "text", "")
    prefix = " · ".join(part for part in (source, section) if part)
    return f"{prefix}\n{text}" if prefix else text


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.counts = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        self.df: Counter[str] = Counter()
        for count in self.counts:
            self.df.update(set(count))
        self.n = len(documents)

    def scores(self, query: str) -> np.ndarray:
        terms = set(tokenize(query))
        output = np.zeros(self.n, dtype=np.float32)
        for index, count in enumerate(self.counts):
            length = self.lengths[index]
            score = 0.0
            for term in terms:
                tf = count.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += (
                    idf
                    * tf
                    * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
                )
            output[index] = score
        return output


@dataclass
class HybridHit:
    chunk: object
    score: float
    vector_rank: int
    bm25_rank: int


class Retriever:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._bm25: BM25 | None = None

    @classmethod
    def load(cls, index_dir: Path | str) -> Retriever:
        return cls(VectorStore.load(index_dir))

    @property
    def bm25(self) -> BM25:
        # 懒加载：纯向量模式不需要建 BM25 索引
        if self._bm25 is None:
            self._bm25 = BM25([embedding_text(chunk) for chunk in self.store.chunks])
        return self._bm25

    def search(
        self,
        query: str,
        top_k: int = 4,
        mode: Mode = "hybrid",
        max_per_source: int = 2,
    ) -> list[SearchHit]:
        vector_scores = self.store.vectors @ embed_query(query)
        vector_order = list(np.argsort(-vector_scores))

        if mode == "vector":
            ranked: list[SearchHit] = []
            per_source: dict[str, int] = {}
            for index in vector_order:
                chunk = self.store.chunks[int(index)]
                key = chunk.citation() if hasattr(chunk, "citation") else chunk.source
                if per_source.get(key, 0) >= max_per_source:
                    continue
                per_source[key] = per_source.get(key, 0) + 1
                ranked.append(SearchHit(chunk, float(vector_scores[int(index)])))  # type: ignore[arg-type]
                if len(ranked) >= top_k:
                    break
            return ranked

        bm25_scores = self.bm25.scores(query)

        # 归一化分数融合：把余弦相似度和 BM25 各自拉伸到 [0, 1]，再等权相加。
        # 为什么不用 RRF：RRF 只看排名，一个"BM25 第 1、向量第 200"的块只能拿
        # 1/(60+0)，会输给"两边都是第 10"的块。对专有名词查询，这是致命的。
        def norm(scores: np.ndarray) -> np.ndarray:
            low, high = float(scores.min()), float(scores.max())
            return (scores - low) / (high - low) if high > low else scores * 0.0

        fused = 0.5 * norm(vector_scores) + 0.5 * norm(bm25_scores)

        ranked = []
        per_source: dict[str, int] = {}
        for index in np.argsort(-fused):
            chunk = self.store.chunks[int(index)]
            key = chunk.citation() if hasattr(chunk, "citation") else chunk.source
            if per_source.get(key, 0) >= max_per_source:
                continue
            per_source[key] = per_source.get(key, 0) + 1
            ranked.append(SearchHit(chunk, float(fused[int(index)])))  # type: ignore[arg-type]
            if len(ranked) >= top_k:
                break
        return ranked


def format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        citation = hit.chunk.citation() if hasattr(hit.chunk, "citation") else hit.chunk.source
        blocks.append(f"[{index}] {citation}\n{hit.chunk.text}")
    return "\n\n".join(blocks)
