"""阶段 5 · prompt caching 实验：同一段前缀，第二次开始便宜一个数量级。

DeepSeek 的上下文缓存是**自动**的：没有 `cache=True` 这种参数。服务端对
请求前缀做匹配，命中部分按缓存价计费，并在 usage 里返回：

    prompt_cache_hit_tokens   命中缓存的 token 数
    prompt_cache_miss_tokens  未命中的 token 数

三个实验：
  A. 固定长前缀 + 5 个不同问题：看第 2~5 次命中多少、便宜多少、快多少；
  B. 每次改前缀开头的一个标记：验证"前缀一变，缓存全废"；
  C. 短前缀：没有可缓存的空间，作为对照。

运行：
  uv run python stage5/caching_demo.py
  uv run python stage5/caching_demo.py --repeat 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from tracing import cost_of  # noqa: E402

load_dotenv(PROJECT / ".env")

MODEL = "deepseek-chat"
RUNTIME = ROOT / "runtime"

# 固定前缀：真实项目里这里会是 system prompt + 工具 schema + few-shot 示例。
# 为了让缓存效果明显，这里放 24 条课程事实，约 2000 token。
FACTS = [
    "CNF 是子句的合取，子句是文字的析取；DIMACS 中每个子句以 0 结尾。",
    "DIMACS 文件头 p cnf n m 中，n 是变量个数，m 是子句个数。",
    "把 A -> B 转成 CNF 得到 ¬A ∨ B；把 A <-> B 转成两条子句。",
    "单元传播：子句只剩一个未赋值文字时，强制该变量取使子句为真的值。",
    "DPLL 的循环是：单元传播、冲突回溯、选择变量分支。",
    "CDCL 在 DPLL 上增加冲突分析与学习子句，以及非时序回溯。",
    "MiniSAT 使用 VSIDS 启发式选择变量，所以两次运行可能给出不同模型。",
    "阻塞子句把当前模型中每个变量取反，恰好排除这一个完整模型。",
    "UNSAT 表示求解器穷尽搜索空间，是对不存在满足赋值的证明。",
    "SMT = SAT + 背景理论，变量可以是整数、位向量、数组等。",
    "用性质否定检查推出关系：结果 unsat 说明性质被推出。",
    "不可满足核是一组足以导致无解的带标签约束，但不保证最小。",
    "8 位无符号位向量中 255 + 1 回绕为 0；有符号解释下 127 + 1 为 -128。",
    "Z3 的 Optimize 在所有合法方案中最小化或最大化目标函数。",
    "Hoare 三元组 {P} C {Q} 中，P 是前置条件，Q 是后置条件。",
    "ACSL 省略 assigns 时默认是 assigns \\everything，不是什么都不改。",
    "WP 从后置条件倒着推，循环必须有 loop invariant 才能继续。",
    "Eva 用抽象解释前向计算可达值，不需要契约，只报告未定义行为告警。",
    "循环不变式要支持初始化、保持、退出三类证明义务。",
    "loop variant 必须非负且每轮严格减小，用来证明循环终止。",
    "空洞证明：前置条件不可满足时，关于这类输入的证明没有实际意义。",
    "WP 的 Unknown 表示没证出来，不代表命题为假，可能是缺不变式或超时。",
    "LTL 的 G/F/X/U 分别表示始终、最终、下一步、直到。",
    "Büchi 接受条件要求无限运行无限次访问接受状态。",
    "实验 1 用 CBMC 的 --signed-overflow-check 检查有符号整数溢出。",
    "__CPROVER_assume 约束非确定输入，排除不满足前提的路径。",
    "中点计算的修正写法是 low + (high - low) / 2，避免先做加法。",
    "普通测试只覆盖实际运行过的输入，CBMC 覆盖所有满足假设的输入。",
    "DIMACS 变量编号从 1 开始，0 是子句结束符而不是变量。",
    "真值表法：每条子句排除一个使公式为假的赋值。",
    "至少两个为真：对每个变量 v，加一条除 v 外其余变量的析取。",
    "至多一个为真：对每一对变量加子句 ¬xi ∨ ¬xj。",
    "MiniSAT 的输出状态有 SATISFIABLE、UNSATISFIABLE、INDETERMINATE。",
    "UNSAT 是穷尽搜索后的证明，INDETERMINATE 只是没算完。",
    "把模型中每个变量取反得到一个阻塞子句，它只排除这一个模型。",
    "两份 CNF 等价，当且仅当在所有完整赋值上满足结果完全相同。",
    "Z3Py 的 Solver.check() 返回 sat、unsat 或 unknown。",
    "Z3 用 Exists 表示存在量词，用 ForAll 表示全称量词。",
    "unsat_core() 返回的标签集合不保证最小，需要自己删减验证。",
    "BitVec 的加法只保留固定位宽，溢出会回绕。",
    "UGT 是无符号比较，位向量的 > 是有符号比较。",
    "Optimize 求最小或最大目标，Solver 只求任一可满足方案。",
    "排课 validSchedule 先检查时间段范围，再检查每一对冲突课程。",
    "不在冲突列表中的课程可以安排在同一个时间段。",
    "ACSL 契约写在 /*@ ... */ 中，单行契约可以写 //@。",
    "requires 是调用者的义务，ensures 是被调函数的义务。",
    "assigns 描述允许修改的内存位置，\\nothing 表示不修改任何内存。",
    "\\valid(a + (0 .. n-1)) 表示这段数组可读可写。",
    "\\old 表示进入函数时的值，\\result 表示函数返回值。",
    "WP 为每个分支和循环边界生成一条证明义务。",
    "WP 的常见状态有 Valid、Invalid、Unknown、Timeout。",
    "Eva 用 Frama_C_interval 指定入口变量的取值范围。",
    "Eva 不证明后置条件，只计算可达值并报告未定义行为告警。",
    "弱后置条件可能让错误实现也通过验证。",
    "规约变异：每次只改一处规约，观察哪条证明义务发生变化。",
    "Rocq 常用 tactic 有 intros、induction、simpl、lia、auto。",
    "Rocq 的 Admitted 表示跳过证明，定理会被当作公理。",
    "LTL 公式在无限轨迹上解释，lasso 轨迹写成 prefix ; loop。",
    "G(req -> F ack) 表示每次请求最终都会得到响应。",
    "检查 M & !P 得到 SAT，说明存在一条违反性质 P 的反例轨迹。",
    "Büchi 自动机接受一条轨迹，当且仅当接受状态被无限次访问。",
    "空洞满足：触发条件从未发生，蕴含性质自动成立。",
]


def build_prefix(nonce: str) -> str:
    lines = [
        f"课程知识卡片（固定前缀，运行标记 {nonce}）",
        "以下事实是回答问题的唯一依据，不要使用卡片以外的知识。",
    ]
    for index, fact in enumerate(FACTS, start=1):
        lines.append(f"[{index:02d}] {fact}")
    lines.append("回答规则：只答一句，不超过 40 字，不要解释。")
    return "\n".join(lines)


QUESTIONS = [
    "CDCL 相比 DPLL 增加了哪两个机制？",
    "阻塞子句的作用是什么？",
    "WP 的 Unknown 是否代表命题为假？",
    "LTL 的 G 和 F 分别表示什么？",
    "循环不变式要支持哪三类证明义务？",
]


def ask(client: OpenAI, system: str, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=80,
    )
    elapsed = time.perf_counter() - started
    usage = response.usage
    prompt = usage.prompt_tokens if usage else 0
    completion = usage.completion_tokens if usage else 0
    hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0) if usage else 0
    miss = int(getattr(usage, "prompt_cache_miss_tokens", prompt - hit) or 0) if usage else prompt
    return {
        "answer": (response.choices[0].message.content or "").strip(),
        "prompt": prompt,
        "completion": completion,
        "hit": hit,
        "miss": miss,
        "seconds": elapsed,
        "cost": cost_of(prompt, completion, hit),
        "uncached_cost": cost_of(prompt, completion, 0),
    }


def print_rows(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'#':>3}{'prompt':>8}{'命中':>8}{'未命中':>9}{'命中率':>8}"
          f"{'耗时':>8}{'费用':>10}{'不缓存费用':>12}")
    for index, row in enumerate(rows, start=1):
        ratio = row["hit"] / row["prompt"] if row["prompt"] else 0
        print(
            f"{index:>3}{row['prompt']:>8}{row['hit']:>8}{row['miss']:>9}"
            f"{ratio:>8.0%}{row['seconds']:>7.2f}s{row['cost']:>10.4f}"
            f"{row['uncached_cost']:>12.4f}"
        )
    total = sum(row["cost"] for row in rows)
    uncached = sum(row["uncached_cost"] for row in rows)
    print(f"    合计：¥{total:.4f}；若完全没有缓存：¥{uncached:.4f}；"
          f"节省 {1 - total / uncached:.0%}")
    cached = [row["seconds"] for row in rows if row["hit"] > 0]
    first = [row["seconds"] for row in rows if row["hit"] == 0]
    if cached:
        message = f"    平均耗时：命中 {sum(cached) / len(cached):.2f}s"
        if first:
            message += f" | 未命中 {sum(first) / len(first):.2f}s"
        print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="prompt caching 实测")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
    )

    nonce = uuid.uuid4().hex[:8]
    long_prefix = build_prefix(nonce)
    questions = (QUESTIONS * args.repeat)[: args.repeat]

    # A. 固定前缀
    rows_a = [ask(client, long_prefix, question) for question in questions]
    print_rows(rows_a, f"A. 固定长前缀（{len(long_prefix)} 字，{args.repeat} 次）")

    # B. 每次改前缀开头的运行标记
    rows_b = [ask(client, build_prefix(uuid.uuid4().hex[:8]), q) for q in questions[:3]]
    print_rows(rows_b, "B. 每次改前缀开头的标记（模拟 system prompt 改动）")

    # C. 短前缀
    short_prefix = "你是课程助教，只答一句，不超过 40 字。"
    rows_c = [ask(client, short_prefix, q) for q in questions[:3]]
    print_rows(rows_c, "C. 短前缀（没有可复用的长前缀）")

    payload = {
        "long_prefix_chars": len(long_prefix),
        "fixed_prefix": rows_a,
        "changed_prefix": rows_b,
        "short_prefix": rows_c,
        "savings": {
            "fixed_total": sum(r["cost"] for r in rows_a),
            "fixed_uncached": sum(r["uncached_cost"] for r in rows_a),
        },
    }
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "caching.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[保存] {RUNTIME / 'caching.json'}")
    print(
        "要点：前缀要稳定且足够长（DeepSeek 以 64 token 为块）；把用户输入、时间戳、\n"
        "随机 ID 放在**后面**。前缀开头改一个字，后面的缓存全部失效。"
    )


if __name__ == "__main__":
    main()
