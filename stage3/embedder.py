"""阶段 3 · 本地 embedding：把文本变成向量。

用 fastembed + `BAAI/bge-small-zh-v1.5`：
  - 512 维，模型约 90MB，CPU 就能跑，不依赖任何 API key；
  - 第一次运行会从 HuggingFace 下载；国内设置 HF_ENDPOINT=https://hf-mirror.com；
  - 文档用 embed()，查询用 query_embed()——bge 的查询侧指令模板会影响召回。

为什么不用 DeepSeek 做 embedding？
  DeepSeek 的 API 只提供 chat/completions，没有 embeddings 接口。
  嵌入模型和生成模型是两类不同的模型，别混为一谈。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DIM = 512
# 不要用临时目录：Windows 重启后临时目录可能被清掉，模型要重新下载
CACHE_DIR = Path.home() / ".cache" / "fastembed"


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    """模型只加载一次。lru_cache 保证同一进程里不会重复加载。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2 归一化。归一化后，点积 = 余弦相似度，省一次除法。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """文档侧 embedding，返回 (n, 512) 的 float32 矩阵，已归一化。"""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    vectors = np.array(
        list(get_model().embed(texts, batch_size=batch_size)),
        dtype=np.float32,
    )
    return normalize(vectors)


def embed_query(text: str) -> np.ndarray:
    """查询侧 embedding，返回 (512,) 向量。

    bge 系列建议查询加指令前缀："为这个句子生成表示以用于检索相关文章："
    fastembed 的 query_embed() 会按模型配置自动加，别手动再拼一遍。
    """
    vector = np.array(
        list(get_model().query_embed([text]))[0],
        dtype=np.float32,
    )
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def embed_one(text: str) -> np.ndarray:
    """把单段文本当作文档 embedding（建索引时用）。"""
    return embed_texts([text])[0]
