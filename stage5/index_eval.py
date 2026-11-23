"""阶段 5 · 消融实验：切块 / top-k / 检索方式，到底哪个有用？

五个配置，一次只改一个变量（尽量）：

  1. vector500_k4   原始基线：500 字切块、纯向量、top-4（stage3/index）
  2. heading500_k4  标题注入：embed 时把“来源 · 章节”拼进正文
  3. heading500_k8  只把 top-k 从 4 提到 8
  4. hybrid500_k6   再加 BM25 + RRF 混合检索
  5. hybrid800_k6   把切块从 500 提到 800

对照组不是“更复杂就更好”，而是用同一套 30 题 eval 说话。
如果某一项没有提升，就如实写“没有提升”。

运行：
  uv run python stage5/index_eval.py --limit 10      # 先小规模试跑
  uv run python stage5/index_eval.py                 # 完整 30 题 × 5 配置

建索引是本地计算（fastembed），不花钱；eval 会调用模型，约 ¥0.6~1.0。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from rag_eval import load_questions, make_client, run_eval  # noqa: E402
from retrieval_v2 import Retriever, embedding_text  # noqa: E402

# retrieval_v2 已经把 stage3 加进 sys.path
from chunking import build_chunks  # noqa: E402
from embedder import embed_texts  # noqa: E402
from rag_store import VectorStore  # noqa: E402

SOURCE = Path("E:/形式化方法")
RUNTIME = ROOT / "runtime"
INDEX_500H = ROOT / "index_500h"   # 标题注入版：embedding 用 section + 正文
INDEX_300 = ROOT / "index_300"     # 小切块：embedding 只用正文
INDEX_800 = ROOT / "index_800"     # 大切块：embedding 只用正文

CONFIGS: list[dict[str, object]] = [
    {
        "label": "vector500_k4",
        "index": PROJECT / "stage3" / "index",
        "top_k": 4,
        "mode": "vector",
        "note": "阶段 3 基线",
    },
    {
        "label": "heading500_k4",
        "index": INDEX_500H,
        "build": (500, 80, True),
        "top_k": 4,
        "mode": "vector",
        "note": "标题注入 embedding（对照）",
    },
    {
        "label": "vector500_k8",
        "index": PROJECT / "stage3" / "index",
        "top_k": 8,
        "mode": "vector",
        "note": "只改 top-k",
    },
    {
        "label": "hybrid500_k6",
        "index": PROJECT / "stage3" / "index",
        "top_k": 6,
        "mode": "hybrid",
        "note": "BM25 混合检索",
    },
    {
        "label": "hybrid800_k6",
        "index": INDEX_800,
        "build": (800, 100, False),
        "top_k": 6,
        "mode": "hybrid",
        "note": "更大切块",
    },
    {
        "label": "hybrid300_k6",
        "index": INDEX_300,
        "build": (300, 50, False),
        "top_k": 6,
        "mode": "hybrid",
        "note": "更小切块",
    },
]


def build_index(size: int, overlap: int, out: Path, heading: bool) -> int:
    if (out / "vectors.npy").exists() and (out / "chunks.jsonl").exists():
        existing = VectorStore.load(out)
        print(f"[索引] 复用 {out}（{len(existing.chunks)} 块）")
        return len(existing.chunks)

    print(f"[索引] 构建 {out}：size={size}, overlap={overlap}, heading={heading}")
    started = time.perf_counter()
    chunks = build_chunks(SOURCE, size=size, overlap=overlap)
    texts = [embedding_text(chunk) if heading else chunk.text for chunk in chunks]
    vectors = embed_texts(texts)
    VectorStore(
        chunks,
        vectors,
        meta={
            "chunk_size": size,
            "overlap": overlap,
            "embedding_text": "source · section + text" if heading else "text",
            "source_root": str(SOURCE),
        },
    ).save(out)
    print(
        f"[索引] {len(chunks)} 块，{time.perf_counter() - started:.1f}s"
        f" → {out}"
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="切块/top-k/检索方式消融实验")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (
        RUNTIME / "index_eval.json"
        if not args.limit
        else RUNTIME / f"index_eval_limit{args.limit}.json"
    )

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    client = make_client()
    results = []
    # 小规模试跑不要覆盖完整结果：所有产物文件名都带 _limitN 后缀。
    suffix = "" if not args.limit else f"_limit{args.limit}"

    for config in CONFIGS:
        index = Path(config["index"])  # type: ignore[arg-type]
        build = config.get("build")
        if isinstance(build, tuple):
            size, overlap, heading = build
            build_index(size, overlap, index, heading)
        retriever = Retriever.load(index)
        summary = run_eval(
            client,
            questions,
            retriever,
            label=str(config["label"]) + suffix,
            top_k=int(config["top_k"]),  # type: ignore[arg-type]
            mode=str(config["mode"]),
            use_judge=not args.no_judge,
        )
        summary["label"] = str(config["label"])
        summary["chunks"] = len(retriever.store.chunks)
        summary["note"] = config.get("note", "")
        results.append(summary)
        (RUNTIME / f"eval_{config['label']}{suffix}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    baseline = results[0]
    print(f"\n=== 消融对比（同一套 {len(questions)} 题）===")
    header = (
        f"{'配置':<18}{'块数':>6}{'k':>3}{'方式':>8}{'kw':>7}{'judge':>7}{'双通过':>8}"
        f"{'Δ':>7}{'平均judge':>10}{'ctx':>7}{'tok':>9}{'¥':>8}{'秒':>7}"
    )
    print(header)
    print("-" * len(header))
    for summary in results:
        delta = (summary["final_pass_rate"] - baseline["final_pass_rate"]) * 100
        print(
            f"{summary['label']:<18}{summary['chunks']:>6}{summary['top_k']:>3}"
            f"{summary['mode']:>8}{summary['keyword_pass_rate']:>7.0%}"
            f"{summary['judge_pass_rate']:>7.0%}{summary['final_pass_rate']:>8.0%}"
            f"{delta:>+7.1f}{summary['avg_judge']:>10.2f}"
            f"{summary['avg_context_coverage']:>7.2f}"
            f"{summary['prompt_tokens'] + summary['completion_tokens']:>9}"
            f"{summary['cost']:>8.3f}{summary['seconds']:>7.1f}"
        )

    payload = {
        "questions": len(questions),
        "results": [
            {key: value for key, value in summary.items() if key != "records"}
            for summary in results
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[保存] {out}")
    if len(questions) < 100:
        print(
            f"注意：样本只有 {len(questions)} 题，差 1 题就是 "
            f"{100 / len(questions):.1f} 个百分点，不要过度解读。"
        )


if __name__ == "__main__":
    main()
