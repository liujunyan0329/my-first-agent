"""阶段 5 · 最小可观测性：span 树 + JSONL + 终端瀑布图。

为什么自己写而不用 Langfuse？
  阶段 5 的第一课不是"装一个平台"，而是先搞清 trace 到底是什么：

      trace  = 一次完整任务（一个 trace_id）
      span   = 任务里的一次操作（一次模型调用、一次工具调用、一次检索）
      parent = 谁调用了谁（span 树）
      属性   = 模型、token、费用、耗时、状态、错误……

  把这三件事亲手实现一遍（本文件约 200 行），再接 Langfuse / OpenTelemetry
  就是"换个导出器"，而不是"学一套新概念"。

本文件提供：
  - Tracer.start(name) -> Tracer          新建一条 trace
  - trace.span(name, kind=..., **attrs)   上下文管理器，自动嵌套、自动计时
  - span.set(**attrs) / span.fail(msg)    记录用量、费用、错误
  - tracer.render()                       终端瀑布图
  - tracer.save_jsonl(path)               落盘（每行一个 span）
  - tracer.to_otel()                      OpenTelemetry / Langfuse 兼容结构

成本口径（DeepSeek，2026-11 账户价目表；换模型请改这三个常量）：
  输入未命中缓存  ¥2 / 百万 token
  输入命中缓存    ¥0.2 / 百万 token     ← 约为未命中的 1/10
  输出           ¥8 / 百万 token
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

PRICE_INPUT_MISS = 2.0 / 1_000_000
PRICE_INPUT_HIT = 0.2 / 1_000_000
PRICE_OUTPUT = 8.0 / 1_000_000


def cost_of(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """按"缓存未命中 / 命中 / 输出"三段计价。

    cached_tokens 是 prompt_tokens 里命中缓存的那部分：
      DeepSeek 返回 prompt_cache_hit_tokens；
      OpenAI 返回 prompt_tokens_details.cached_tokens。
    """
    hit = min(max(cached_tokens, 0), max(prompt_tokens, 0))
    miss = max(prompt_tokens, 0) - hit
    return (
        miss * PRICE_INPUT_MISS
        + hit * PRICE_INPUT_HIT
        + max(completion_tokens, 0) * PRICE_OUTPUT
    )


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    start: float                       # perf_counter 秒
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""

    @property
    def duration_ms(self) -> float:
        end = self.end if self.end is not None else time.perf_counter()
        return (end - self.start) * 1000

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error[:500]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start_ms": round((self.start - _TRACE_ORIGIN.get(self.trace_id, self.start)) * 1000, 3),
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error": self.error,
            "attrs": self.attrs,
        }


# 记录每条 trace 的起点，用于把 span 时间换算成"相对 trace 开始"的毫秒数。
_TRACE_ORIGIN: dict[str, float] = {}


class Tracer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.trace_id = uuid.uuid4().hex[:16]
        self.started = time.perf_counter()
        # perf_counter 是"开机以来的秒数"，不是 Unix 时间；OTel 需要 epoch 纳秒。
        self.wall_started = time.time()
        _TRACE_ORIGIN[self.trace_id] = self.started
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attrs: Any) -> Iterator[Span]:
        parent = self._stack[-1].span_id if self._stack else None
        item = Span(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent,
            name=name,
            kind=kind,
            start=time.perf_counter(),
            attrs=dict(attrs),
        )
        self.spans.append(item)
        self._stack.append(item)
        try:
            yield item
        except Exception as exc:                     # noqa: BLE001 —— trace 必须记录所有异常
            item.fail(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._stack.pop()
            item.end = time.perf_counter()

    # ------------------------------ 汇总 ------------------------------
    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    @property
    def cost(self) -> float:
        return sum(float(span.attrs.get("cost", 0.0)) for span in self.spans)

    @property
    def tokens(self) -> int:
        return sum(
            int(span.attrs.get("prompt_tokens", 0))
            + int(span.attrs.get("completion_tokens", 0))
            for span in self.spans
        )

    def roots(self) -> list[Span]:
        ids = {span.span_id for span in self.spans}
        return [span for span in self.spans if span.parent_id not in ids]

    def children(self, parent: Span) -> list[Span]:
        return [span for span in self.spans if span.parent_id == parent.span_id]

    # ------------------------------ 展示 ------------------------------
    def render(self, width: int = 34) -> str:
        """终端瀑布图。每一行：名字 + 相对开始 + 条形 + 耗时 + 关键属性。"""
        if not self.spans:
            return "（空 trace）"
        total = max(self.total_ms, 1.0)
        lines = [
            f"trace {self.trace_id}  {self.name}  总耗时 {total / 1000:.2f}s  "
            f"{len(self.spans)} spans  {self.tokens} tok  ¥{self.cost:.4f}"
        ]

        def walk(item: Span, depth: int) -> None:
            offset = (item.start - self.started) * 1000
            left = int(offset / total * width)
            size = max(1, int(item.duration_ms / total * width))
            size = min(size, width - left)
            bar = "·" * left + "█" * size
            bar = bar.ljust(width, " ")
            label = "  " * depth + item.name
            extra = _summary(item)
            mark = "✗" if item.status == "error" else " "
            lines.append(
                f"{mark} {label:<40} {bar} {item.duration_ms / 1000:>6.2f}s {extra}"
            )
            if item.status == "error" and item.error:
                lines.append("  " * (depth + 2) + f"  ↳ {item.error}")
            for child in self.children(item):
                walk(child, depth + 1)

        for root in self.roots():
            walk(root, 0)
        return "\n".join(lines)

    # ------------------------------ 导出 ------------------------------
    def save_jsonl(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for span in self.spans:
                file.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")
        return path

    def to_otel(self) -> list[dict[str, Any]]:
        """转成 OpenTelemetry 风格的 span 列表。

        Langfuse / Phoenix / Jaeger 都能接 OTel；字段名按 GenAI 语义约定：
        gen_ai.system / gen_ai.request.model / gen_ai.usage.input_tokens ...
        """
        base = int(self.wall_started * 1_000_000_000)
        output: list[dict[str, Any]] = []
        for span in self.spans:
            attrs: dict[str, Any] = {
                "gen_ai.operation.name": span.kind,
                **{f"app.{key}": value for key, value in span.attrs.items()},
            }
            mapping = {
                "model": "gen_ai.request.model",
                "prompt_tokens": "gen_ai.usage.input_tokens",
                "completion_tokens": "gen_ai.usage.output_tokens",
                "cached_tokens": "gen_ai.usage.cached_tokens",
            }
            for key, target in mapping.items():
                if key in span.attrs:
                    attrs[target] = span.attrs[key]
            output.append(
                {
                    "traceId": span.trace_id,
                    "spanId": span.span_id,
                    "parentSpanId": span.parent_id or "",
                    "name": span.name,
                    "kind": span.kind,
                    "startTimeUnixNano": base + int((span.start - self.started) * 1e9),
                    "endTimeUnixNano": base
                    + int(((span.end or span.start) - self.started) * 1e9),
                    "status": {"code": "ERROR" if span.status == "error" else "OK"},
                    "attributes": attrs,
                }
            )
        return output

    def save_otel(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_otel(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def _summary(span: Span) -> str:
    parts: list[str] = []
    if "model" in span.attrs:
        parts.append(str(span.attrs["model"]))
    if "prompt_tokens" in span.attrs:
        parts.append(
            f"{span.attrs.get('prompt_tokens', 0)}/{span.attrs.get('completion_tokens', 0)} tok"
        )
        if span.attrs.get("cached_tokens"):
            parts.append(f"缓存 {span.attrs['cached_tokens']}")
    if span.attrs.get("cost"):
        parts.append(f"¥{float(span.attrs['cost']):.4f}")
    if "tool" in span.attrs:
        parts.append(str(span.attrs["tool"]))
    if span.status == "error":
        parts.append("ERROR")
    return "  ".join(parts)
