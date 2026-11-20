"""阶段 3 · 上下文管理器：滑动窗口 + 摘要压缩 + token 预算。

阶段 1/2 的 agent 只会往 self.messages 后面 append，历史无限增长。
这个模块把"哪些消息发给模型"从"对话记录"里拆了出来：

    self.messages       ← 完整历史（永久保留，便于人类复盘）
    prepare() 的返回值  ← 本次真正发给模型的部分（受预算约束）

三种策略：
  none     全量发送。量小的时候最简单，量一大就超限 + 烧钱。
  window   只保留最近 N 轮。便宜，但早期信息直接丢失。
  summary  丢弃的旧轮次先用模型压缩成摘要，摘要常驻上下文。

核心难点不是"删消息"，而是**不能删坏消息序列**：
OpenAI 协议要求每条 assistant 的 tool_calls 后面必须跟对应 tool 消息，
从中间切一刀会直接 400。

运行演示：
    uv run python stage3/long_convo_demo.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Message = dict[str, Any]
Strategy = Literal["none", "window", "summary"]

SUMMARY_SYSTEM = """\
你在压缩一段 agent 对话的历史记录，供后续轮次继续使用。
必须保留：
1. 用户的目标、约束和偏好（原话中的专有名词、代号、数字、路径要逐字保留）；
2. 已经确认的事实；
3. 已经做出的决定；
4. 尚未解决的问题。
可以删除：寒暄、重复、模型的客套话、与结论无关的中间过程。
用中文要点式输出，不超过 200 字。只输出摘要本身。"""


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # 基本汉字
        or 0x3000 <= code <= 0x303F   # 中文标点
        or 0xFF00 <= code <= 0xFFEF   # 全角字符
    )


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    DeepSeek 没有公开本地 tokenizer，而且不同模型的分词器不同。
    经验值：中文 1 字 ≈ 1 token，英文/数字 ≈ 4 字符 1 token。
    这个估算偏保守（会比真实值大 10%~20%），对"设预算"来说这是好事：
    宁可早一点压缩，也不要贴着上限被服务端拒绝。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return int(cjk + other / 4) + 1


def message_tokens(message: Message) -> int:
    """一条消息的 token 估算：文本 + tool_calls 的 JSON + 角色/分隔符开销。"""
    total = 4  # role、分隔符等固定开销
    content = message.get("content")
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    total += estimate_tokens(text)
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        total += estimate_tokens(str(function.get("name", "")))
        total += estimate_tokens(str(function.get("arguments", "")))
    return total


def messages_tokens(messages: list[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


def flatten(message: Message) -> str:
    """把一条消息压成一行文本，只用于给摘要模型看。"""
    role = message.get("role", "?")
    content = message.get("content")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    parts = [f"{role}: {text}".strip()]
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        parts.append(
            f"assistant 调用工具 {function.get('name')}"
            f"({function.get('arguments')})"
        )
    return "\n".join(part for part in parts if part.strip())


def _turn_groups(history: list[Message]) -> list[list[Message]]:
    """按"用户发言"把历史切成组。

    一组 = 一条 user 消息 + 它引发的所有 assistant/tool 消息。
    删除历史时必须以组为单位，否则会留下"没有问题的工具结果"
    或者"没有回答的用户问题"这种残缺序列。
    """
    groups: list[list[Message]] = []
    for message in history:
        if message.get("role") == "user" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def _clean_summary(text: str) -> str:
    """模型偶尔会加"以下是摘要："之类的前缀，削掉。"""
    text = text.strip()
    for prefix in ("以下是摘要：", "摘要：", "对话摘要："):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


@dataclass
class ContextStats:
    strategy: Strategy = "none"
    full_tokens: int = 0          # 完整历史估算
    sent_tokens: int = 0          # 实际发送估算（含摘要）
    dropped_messages: int = 0     # 已被移出窗口的消息数
    summary_calls: int = 0        # 摘要调用次数
    summary_prompt_tokens: int = 0
    summary_completion_tokens: int = 0

    @property
    def summary_cost_tokens(self) -> int:
        return self.summary_prompt_tokens + self.summary_completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "full_tokens": self.full_tokens,
            "sent_tokens": self.sent_tokens,
            "dropped_messages": self.dropped_messages,
            "summary_calls": self.summary_calls,
            "summary_prompt_tokens": self.summary_prompt_tokens,
            "summary_completion_tokens": self.summary_completion_tokens,
        }


@dataclass
class ContextManager:
    """维护完整历史，并按预算产出"本次发给模型"的消息列表。

    用法：
        ctx = ContextManager(system_prompt, budget=6000, strategy="summary")
        ctx.add({"role": "user", "content": "..."})
        messages = ctx.prepare(client)      # 可能触发一次摘要调用
        agent.messages = messages           # 发给模型
    """

    system_prompt: str
    budget: int = 6000
    strategy: Strategy = "summary"
    keep_last_turns: int = 3          # window 策略至少保留最近几轮
    model: str = "deepseek-chat"
    history: list[Message] = field(default_factory=list)
    summary: str = ""
    cursor: int = 0                   # history[0:cursor] 已被折叠进摘要/窗口
    stats: ContextStats = field(default_factory=ContextStats)

    def __post_init__(self) -> None:
        self.stats.strategy = self.strategy

    # ------------------------------ 写入 ------------------------------
    def add(self, message: Message) -> None:
        self.history.append(message)

    # ------------------------------ 读取 ------------------------------
    def _summary_message(self) -> list[Message]:
        if not self.summary:
            return []
        return [{
            "role": "system",
            "content": f"## 早期对话摘要（由系统压缩，事实仍然有效）\n{self.summary}",
        }]

    def visible(self) -> list[Message]:
        """不调用模型，只看当前窗口（用于打印/测试）。"""
        return (
            [{"role": "system", "content": self.system_prompt}]
            + self._summary_message()
            + self.history[self.cursor:]
        )

    def prepare(self, client: Any) -> list[Message]:
        """按预算生成发给模型的消息列表，必要时压缩或丢弃旧轮次。"""
        self.stats.full_tokens = (
            estimate_tokens(self.system_prompt)
            + messages_tokens(self.history)
        )

        if self.strategy == "none":
            self.stats.sent_tokens = self.stats.full_tokens
            return self.visible()

        if self.strategy == "window":
            self._shrink_window()
        else:
            self._shrink_summary(client)

        visible = self.visible()
        self.stats.sent_tokens = messages_tokens(visible)
        return visible

    # ------------------------------ 内部 ------------------------------
    def _over_budget(self) -> bool:
        return messages_tokens(self.visible()) > self.budget

    def _shrink_window(self) -> None:
        """丢弃最旧的整组，直到进入预算；至少保留 keep_last_turns 组。"""
        groups = _turn_groups(self.history[self.cursor:])
        # 最后一组可能是"当前这一问"，无论如何保留
        while self._over_budget() and len(groups) > self.keep_last_turns:
            dropped = groups.pop(0)
            self.cursor += len(dropped)
            self.stats.dropped_messages += len(dropped)

    def _shrink_summary(self, client: Any) -> None:
        """把最旧的若干整组抽出来，交给模型压缩进滚动摘要。

        注意：不是"丢一条摘要一条"——那样 40 轮对话会产生 40 次摘要调用，
        比直接发全文还贵。正确做法是**先算出要丢多少组，再分批压缩**。
        """
        while self._over_budget():
            groups = _turn_groups(self.history[self.cursor:])
            if len(groups) <= 1:
                break  # 只剩当前这一问，不能再删

            overhead = (
                estimate_tokens(self.system_prompt)
                + estimate_tokens(self.summary)
                + 50
            )
            group_tokens = [messages_tokens(group) for group in groups]
            total = overhead + sum(group_tokens)

            # 多删组，直到剩余内容 + 未来摘要（预留 300 tok）进入预算
            drop_count = 0
            while (
                total + 300 > self.budget
                and drop_count < len(groups) - 1
            ):
                total -= group_tokens[drop_count]
                drop_count += 1
            if drop_count == 0:
                break

            dropped = [m for group in groups[:drop_count] for m in group]
            self._fold_batches(client, dropped)
            self.cursor += len(dropped)
            self.stats.dropped_messages += len(dropped)

    def _fold_batches(self, client: Any, messages: list[Message]) -> None:
        """把待丢消息按约 6000 tok 一批压缩，避免单次输入过大。"""
        batch: list[Message] = []
        batch_tokens = 0
        for message in messages:
            size = message_tokens(message)
            if batch and batch_tokens + size > 6000:
                self._fold_into_summary(client, batch)
                batch, batch_tokens = [], 0
            batch.append(message)
            batch_tokens += size
        if batch:
            self._fold_into_summary(client, batch)

    def _fold_into_summary(self, client: Any, messages: list[Message]) -> None:
        transcript = "\n".join(flatten(m) for m in messages)
        prompt = (
            f"已有摘要：\n{self.summary or '（无）'}\n\n"
            f"新增对话：\n{transcript}\n\n"
            "把两者合并成一份新的摘要。"
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        self.stats.summary_calls += 1
        if response.usage is not None:
            self.stats.summary_prompt_tokens += response.usage.prompt_tokens
            self.stats.summary_completion_tokens += response.usage.completion_tokens
        content = response.choices[0].message.content or ""
        self.summary = _clean_summary(content) or self.summary
