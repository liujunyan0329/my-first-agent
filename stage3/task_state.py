"""阶段 3 · 任务状态（task state）提取：把对话中的关键事实抽成结构化 JSON。

滑动窗口会丢信息，摘要会损失细节，而且两者都是"不可查询的自然语言"。
任务状态是另一条路：每几轮让模型把事实抽成结构化字段，之后只带这些字段。

    原始对话（不断增长）  →  每 N 轮提取一次  →  TaskState（有界、可读、可 diff）

代价：每次提取是一次额外的模型调用；状态本身也会过时（需要覆盖规则）。
收益：上下文占用从"随轮数增长"变成"固定上限"，而且下一阶段做评估、
多 agent 协作时，结构化状态比一段摘要更好用。

运行演示：
    uv run python stage3/task_state.py
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

Message = dict[str, Any]

EXTRACT_SYSTEM = """\
你是对话状态提取器。读取"已有状态"和"新增对话"，输出更新后的状态 JSON，格式固定为：
{
  "goal": "用户当前要完成的目标，一句话；没有就空字符串",
  "facts": ["能独立读懂的短事实，例如：项目代号是青鸟"],
  "decisions": ["已经确定的做法，例如：用 SQLite 而不是 Postgres"],
  "open_questions": ["尚未解决的问题"]
}
规则：
1. 只提取对话中明确出现的信息，不要推测、不要补全。
2. 专有名词、数字、路径、代号逐字保留。
3. 已经有的事实不要重复；如果新对话推翻了旧事实，用新事实替换旧的。
4. 每类最多 8 条，优先保留仍然相关的内容。
5. 只输出 JSON，不要任何解释。"""


class TaskState(BaseModel):
    """有界的结构化状态。类本身就是 schema，不手写解析逻辑。"""

    goal: str = ""
    facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    def merge(self, newer: TaskState) -> TaskState:
        """新状态覆盖旧状态：goal 用新的（非空时），列表去重后合并。"""
        return TaskState(
            goal=newer.goal or self.goal,
            facts=_merge_list(self.facts, newer.facts),
            decisions=_merge_list(self.decisions, newer.decisions),
            open_questions=_merge_list(self.open_questions, newer.open_questions),
        )

    def render(self) -> str:
        """注入系统提示用的文本形态。注意：只放结构化字段，不放原始对话。"""
        lines = ["## 任务状态（系统维护，事实仍然有效）"]
        if self.goal:
            lines.append(f"目标：{self.goal}")
        for title, items in (
            ("已知事实", self.facts),
            ("已定决策", self.decisions),
            ("待解决问题", self.open_questions),
        ):
            if items:
                lines.append(f"{title}：")
                lines.extend(f"- {item}" for item in items)
        return "\n".join(lines)

    def token_estimate(self) -> int:
        from context import estimate_tokens

        return estimate_tokens(self.render())


def _normalize(text: str) -> str:
    """去重用的归一化：去掉空白和标点，只留下字。"""
    return re.sub(r"[\s，。！？、；：,.!?;:（）()]", "", text).lower()


def _merge_list(old: list[str], new: list[str], limit: int = 8) -> list[str]:
    """保留顺序去重；旧条目在前，新条目覆盖语义完全相同的旧条目。"""
    result: list[str] = []
    seen: set[str] = set()
    # 新条目优先保留：旧条目里和新条目重复的会被跳过
    for item in [*new, *old]:
        key = _normalize(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result[:limit]


class StateExtractor:
    """每调用一次 extract()，就产生一次小模型调用。"""

    def __init__(self, client: Any, model: str = "deepseek-chat") -> None:
        self.client = client
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def extract(
        self,
        new_messages: list[Message],
        previous: TaskState | None = None,
        max_retries: int = 1,
    ) -> TaskState:
        previous = previous or TaskState()
        transcript = "\n".join(_flatten(m) for m in new_messages)
        user_prompt = (
            f"已有状态：\n{previous.model_dump_json(indent=2)}\n\n"
            f"新增对话：\n{transcript}\n\n"
            "输出更新后的完整状态 JSON。"
        )

        last_error = ""
        for _ in range(max_retries + 1):
            self.calls += 1
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": user_prompt
                     + (f"\n\n上次输出有问题：{last_error}" if last_error else "")},
                ],
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            if response.usage is not None:
                self.prompt_tokens += response.usage.prompt_tokens
                self.completion_tokens += response.usage.completion_tokens

            content = response.choices[0].message.content or ""
            try:
                state = TaskState.model_validate_json(content)
            except (ValidationError, ValueError) as exc:
                last_error = f"JSON 不符合约定格式：{exc}"
                continue
            # 模型可能把旧状态也原样抄回来，这里再合并一次，保证不丢字段
            return previous.merge(state)

        # 两次都失败：放弃本轮提取，保留旧状态。不要用坏 JSON 污染状态。
        return previous


def _flatten(message: Message) -> str:
    role = message.get("role", "?")
    content = message.get("content")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return f"{role}: {text}"


# ------------------------------ 演示 ------------------------------
DEMO_CONVERSATION: list[list[Message]] = [
    # 第 1 个检查点
    [
        {"role": "user", "content":
         "我们要给实验 2 的 SAT 求解器做一个 agent 前端，项目代号先叫“青鸟”。"
         "我的环境是 Windows 11 + MSYS2，C++17。"},
        {"role": "assistant", "content":
         "收到。项目“青鸟”的目标是为 SAT 求解器做 agent 前端；环境 Windows 11 + MSYS2，C++17。"},
        {"role": "user", "content":
         "求解器本身用 C++，不要重写；agent 层用 Python 调它的命令行。"},
        {"role": "assistant", "content":
         "决定：求解器保持 C++ 实现，agent 层用 Python 通过 CLI 调用。"},
    ],
    # 第 2 个检查点
    [
        {"role": "user", "content":
         "输入格式就用 DIMACS CNF，路径在 E:/形式化方法/lab2/。LLM 先用 deepseek-chat。"},
        {"role": "assistant", "content":
         "已记录：输入 DIMACS CNF，数据目录 E:/形式化方法/lab2/，模型 deepseek-chat。"},
        {"role": "user", "content":
         "另外别用 LangChain，我想继续用手写的循环。"},
        {"role": "assistant", "content":
         "决定：不引入 LangChain，继续手写 agent 循环。"},
    ],
    # 第 3 个检查点
    [
        {"role": "user", "content":
         "预算控制在每天 1 元以内。还没想好的是结果怎么做可视化。"},
        {"role": "assistant", "content":
         "约束：每日成本 ≤ 1 元。未决事项：结果可视化方案。"},
        {"role": "user", "content":
         "对了，把“青鸟”改名成“蓝鹊”。"},
        {"role": "assistant", "content":
         "已更新：项目代号由“青鸟”改为“蓝鹊”。"},
    ],
]


def main() -> None:
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    extractor = StateExtractor(client)

    state = TaskState()
    print("=== 每 4 条消息提取一次状态 ===\n")
    for index, batch in enumerate(DEMO_CONVERSATION, start=1):
        state = extractor.extract(batch, state)
        print(f"--- 检查点 {index}（新增 {len(batch)} 条，累计调用 {extractor.calls} 次）---")
        print(state.model_dump_json(indent=2))
        print(f"状态注入成本：约 {state.token_estimate()} tok\n")

    print("=== 最终注入系统的文本 ===\n")
    print(state.render())
    print(
        f"\n[用量] 提取调用 {extractor.calls} 次，"
        f"{extractor.prompt_tokens + extractor.completion_tokens} tok"
    )
    print(f"[状态大小] {state.token_estimate()} tok（与对话轮数无关的有界值）")

    out = Path(__file__).parent / "runtime" / "task_state.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    print(f"[已保存] {out}")


if __name__ == "__main__":
    main()
