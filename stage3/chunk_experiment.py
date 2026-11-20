"""阶段 3 · 实验三：切块大小和 top-k 到底怎么选？用数据说话。

只测**检索**，不调用生成模型：
  对每组 (chunk_size, top_k)，看标准答案的关键词有没有出现在召回的块里。
这样可以便宜、快速、可重复地比较参数，避免"感觉 500 字比较好"。

指标：
  Recall@k —— 12 道题里，答案关键词被 top-k 块覆盖的比例（0~1）。
  它比"端到端答对率"更纯粹：排除模型写作能力的干扰。

运行：
    uv run python stage3/chunk_experiment.py
    uv run python stage3/chunk_experiment.py --sizes 300 500 800 --top-ks 2 4 6
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from chunking import build_chunks
from embedder import embed_query, embed_texts
from rag_store import VectorStore

SOURCE_ROOT = Path("E:/形式化方法")

# (查询, ((同义词组...), ...)) —— 和 compare_rag.py 的思路一致
QUERIES: list[tuple[str, tuple[tuple[str, ...], ...]]] = [
    ("实验 2 使用哪个 SAT 求解器", (("MiniSAT", "minisat"),)),
    ("实验 3 使用哪个 SMT 求解器", (("Z3", "z3"),)),
    ("实验 4 ACSL 规约验证平台", (("Frama-C", "Frama_C", "frama"),)),
    ("实验 4 交互式定理证明工具", (("Rocq", "Coq", "rocq", "coq"),)),
    ("实验 5 使用的 LTL 与自动机工具", (("Spot", "spot"), ("Aalta", "aalta"))),
    ("Büchi 自动机 基于状态 接受标记", (("基于状态", "状态标记"),)),
    ("单元子句的定义", (("只含一个文字", "只包含一个文字", "只有一个文字"),)),
    (
        "循环不变式任务 索引范围 计数范围 终止度量",
        (("索引范围",), ("计数范围",), ("终止度量", "变式")),
    ),
    ("DPLL 与 CDCL 的区别", (("DPLL",), ("CDCL",))),
    ("Aalta 检查 LTL 可满足性", (("Aalta",), ("可满足",))),
    ("Frama-C 最弱前置条件 WP 插件", (("最弱前置条件", "Weakest Precondition", "WP"),)),
    ("Spot 把 LTL 公式转换为 Büchi 自动机", (("Büchi",), ("Spot",))),
]


def coverage(text: str, groups: tuple[tuple[str, ...], ...]) -> float:
    lowered = text.lower()
    hits = sum(
        1 for group in groups
        if any(keyword.lower() in lowered for keyword in group)
    )
    return hits / len(groups)


def evaluate(store: VectorStore, top_k: int) -> tuple[float, float]:
    scores: list[float] = []
    top_scores: list[float] = []
    for query, groups in QUERIES:
        hits = store.search(embed_query(query), top_k=top_k, max_per_source=2)
        if not hits:
            scores.append(0.0)
            continue
        joined = "\n".join(hit.chunk.text for hit in hits)
        scores.append(coverage(joined, groups))
        top_scores.append(hits[0].score)
    return sum(scores) / len(scores), sum(top_scores) / max(len(top_scores), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 500, 800])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--overlap-ratio", type=float, default=0.16,
                        help="重叠字数 = size × ratio")
    args = parser.parse_args()

    print("=== 实验 A：不同切块大小，Recall@4 ===")
    print(f"{'size':>6}{'overlap':>9}{'块数':>7}{'平均字数':>9}"
          f"{'Recall@4':>10}{'top1相似度':>11}{'耗时s':>8}")
    stores: dict[int, VectorStore] = {}
    for size in args.sizes:
        overlap = max(20, int(size * args.overlap_ratio))
        started = time.perf_counter()
        chunks = build_chunks(SOURCE_ROOT, size=size, overlap=overlap)
        vectors = embed_texts([chunk.text for chunk in chunks])
        elapsed = time.perf_counter() - started
        store = VectorStore(chunks, vectors)
        stores[size] = store
        recall, top_score = evaluate(store, top_k=4)
        avg_chars = sum(len(chunk.text) for chunk in chunks) / len(chunks)
        print(f"{size:>6}{overlap:>9}{len(chunks):>7}{avg_chars:>9.0f}"
              f"{recall:>10.0%}{top_score:>11.3f}{elapsed:>8.1f}")

    best_size = max(args.sizes)
    print(f"\n=== 实验 B：切块 {best_size}，不同 top-k ===")
    print(f"{'top_k':>6}{'Recall':>9}{'送进上下文的字数':>18}")
    store = stores[best_size]
    for top_k in args.top_ks:
        scores = []
        chars = 0
        for query, groups in QUERIES:
            hits = store.search(embed_query(query), top_k=top_k, max_per_source=2)
            joined = "\n".join(hit.chunk.text for hit in hits)
            chars += len(joined)
            scores.append(coverage(joined, groups))
        print(f"{top_k:>6}{sum(scores) / len(scores):>9.0%}"
              f"{chars / len(QUERIES):>18.0f}")

    print("\n解读：")
    print("- 块太小 → 每块信息不完整，答案被切碎；块太大 → 一块混多个主题，")
    print("  embedding 被平均，top-k 里挤进无关内容，还浪费上下文预算。")
    print("- top-k 越大召回越高，但送进上下文的字数和噪声也线性增长；")
    print("  找到拐点后就停，别盲目取 10。")


if __name__ == "__main__":
    main()
