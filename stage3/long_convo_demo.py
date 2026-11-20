"""阶段 3 · 实验一：40 轮对话后，三种上下文策略还记不记得开头。

做法（控制变量，避免烧钱）：
  - 前 3 轮埋下关键事实（项目代号、部署城市、回答偏好）；
  - 后面 40 轮是本地生成的"工作记录"，不调用模型，只用来把历史撑大；
  - 最后用同一个问题问三种策略，比较"发送 token / 摘要成本 / 是否答对"。

为什么不用模型生成 40 轮？
  那样既慢又贵，而且会引入随机性。这个实验要考察的是**上下文管理策略**，
  不是模型聊天能力，所以填充内容用模板生成就够了。

运行：
    uv run python stage3/long_convo_demo.py
    uv run python stage3/long_convo_demo.py --turns 60 --budget 2500
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from context import ContextManager, estimate_tokens, messages_tokens

MODEL = "deepseek-chat"
PRICE_INPUT = 2.0 / 1_000_000
PRICE_OUTPUT = 8.0 / 1_000_000

SYSTEM_PROMPT = (
    "你是一个项目助理。请只根据对话中出现过的信息回答；"
    "如果对话里没有，就明确说“对话中没有提到”，不要猜。"
)

SEED_TURNS = [
    ("我的项目代号叫蓝鹊，请记住。", "好的，项目代号：蓝鹊。"),
    ("项目部署在上海机房，我在杭州办公。", "已记录：部署在上海，你在杭州。"),
    ("以后回答我先给结论，别长篇大论。", "明白，先给结论。"),
]

FILLER_USERS = [
    "第 {i} 轮：继续优化 CNF 解析器，今天处理了嵌套括号、空子句和注释行这几种边界情况，"
    "把递归下降改成显式栈之后，性能从 12ms 降到 9ms；顺手补了 3 个单元测试，"
    "分别覆盖空文件、只有注释的文件、以及变量编号不连续的文件。下一步想看看"
    "大文件（十万子句量级）下的内存分配是否成为瓶颈。",
    "第 {i} 轮：把 DPLL 的单元传播改成迭代实现，避免了深递归导致的栈溢出。"
    "在随机 3-SAT 上测了 200 个样例：50 变量的平均耗时下降 18%，100 变量下降 9%，"
    "但 200 变量上反而慢了 2%，怀疑是缓存不友好。今天先记录现象，"
    "等做完性能剖析再决定是否回退这部分改动。",
    "第 {i} 轮：对比了 watched literals 和朴素扫描两种策略。前者在小规模公式上"
    "优势不明显，甚至因为维护 watch 列表多了常数开销；但在 500 变量以上明显更快，"
    "冲突传播次数大约少一半。测试脚本放在 bench 目录，跑一次全量要 6 分钟，"
    "所以只保留了 20 个有代表性的种子。",
    "第 {i} 轮：整理实验日志，发现回溯次数和变量启发式关系很大。"
    "试了 VSIDS 的简化版：只按最近冲突频率排序，不做衰减。冲突数减少约四分之一，"
    "但在对称性强的图上没有改善。顺便把日志格式从纯文本改成一行一个 JSON，"
    "方便后面用脚本画图。",
    "第 {i} 轮：检查求解器的内存占用。把子句数据库从字典改成连续数组之后，"
    "峰值内存下降 15%，但代码可读性变差了，而且删除子句时需要墓碑标记。"
    "今天先写了一份设计文档，列出三种方案的权衡：数组+墓碑、分代存储、"
    "以及保持字典但换用更紧凑的键。",
]

FILLER_ASSISTANTS = [
    "收到，记录第 {i} 轮进展。",
    "已记录：第 {i} 轮的优化与测试结果。",
    "明白，第 {i} 轮内容已归档。",
]


def build_history(turns: int) -> list[dict[str, Any]]:
    """生成不含最终提问的历史：3 个种子轮 + turns 个填充轮。"""
    history: list[dict[str, Any]] = []
    for user_text, assistant_text in SEED_TURNS:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
    for i in range(1, turns + 1):
        user_text = FILLER_USERS[(i - 1) % len(FILLER_USERS)].format(i=i)
        assistant_text = FILLER_ASSISTANTS[(i - 1) % len(FILLER_ASSISTANTS)].format(i=i)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
    return history


QUESTION = "请只回答两个问题：我的项目代号是什么？项目部署在哪个城市？"


@dataclass
class Result:
    strategy: str
    sent_tokens: int
    summary_calls: int
    summary_tokens: int
    answer: str
    prompt_tokens: int
    completion_tokens: int
    summary_prompt_tokens: int
    summary_completion_tokens: int
    seconds: float

    @property
    def cost(self) -> float:
        # 摘要调用也是钱：不把它算进来，summary 策略会显得"又便宜又记得住"
        return (
            (self.prompt_tokens + self.summary_prompt_tokens) * PRICE_INPUT
            + (self.completion_tokens + self.summary_completion_tokens) * PRICE_OUTPUT
        )


def run_strategy(
    client: OpenAI,
    strategy: str,
    history: list[dict[str, Any]],
    budget: int,
) -> Result:
    ctx = ContextManager(
        system_prompt=SYSTEM_PROMPT,
        budget=budget,
        strategy=strategy,  # type: ignore[arg-type]
        keep_last_turns=3,
        model=MODEL,
    )
    for message in history:
        ctx.add(message)
    ctx.add({"role": "user", "content": QUESTION})

    started = time.perf_counter()
    prepared = ctx.prepare(client)  # summary 策略会在这里调用模型

    response = client.chat.completions.create(
        model=MODEL,
        messages=cast(list[ChatCompletionMessageParam], prepared),
        temperature=0.0,
        max_tokens=120,
    )
    elapsed = time.perf_counter() - started
    usage = response.usage
    return Result(
        strategy=strategy,
        sent_tokens=messages_tokens(prepared),
        summary_calls=ctx.stats.summary_calls,
        summary_tokens=ctx.stats.summary_cost_tokens,
        answer=(response.choices[0].message.content or "").replace("\n", " "),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        summary_prompt_tokens=ctx.stats.summary_prompt_tokens,
        summary_completion_tokens=ctx.stats.summary_completion_tokens,
        seconds=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--budget", type=int, default=3000)
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    history = build_history(args.turns)
    full_tokens = estimate_tokens(SYSTEM_PROMPT) + messages_tokens(history)
    print(
        f"历史：3 个种子轮 + {args.turns} 个填充轮，共 {len(history)} 条消息，"
        f"约 {full_tokens} tok"
    )
    print(f"预算：{args.budget} tok；最终问题：{QUESTION}\n")

    results: list[Result] = []
    for strategy in ("none", "window", "summary"):
        print(f"运行策略 {strategy} ...")
        result = run_strategy(client, strategy, history, args.budget)
        results.append(result)
        print(f"  答：{result.answer}")
        if strategy == "summary":
            # 展示摘要实际长什么样（重新算一次代价高，这里只提示位置）
            pass

    print("\n=== 对比 ===")
    header = f"{'策略':<8}{'发送tok':>8}{'摘要调用':>8}{'摘要tok':>8}{'费用¥':>8}{'秒':>6}  回答"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.strategy:<8}{r.sent_tokens:>8}{r.summary_calls:>8}"
            f"{r.summary_tokens:>8}{r.cost:>8.4f}{r.seconds:>6.1f}  {r.answer}"
        )
    print("（费用已包含摘要调用的输入 + 输出 token）")

    correct = {
        "none": "蓝鹊" in results[0].answer,
        "window": "蓝鹊" in results[1].answer,
        "summary": "蓝鹊" in results[2].answer,
    }
    print("\n=== 是否还记得“蓝鹊” ===")
    for strategy, ok in correct.items():
        print(f"{strategy:<8}{'✅ 记得' if ok else '❌ 忘了/答错'}")


if __name__ == "__main__":
    main()
