"""阶段 2 · 一个最小的 MCP server（stdio transport）。

它把 3 个工具、1 个 resource、1 个 prompt 暴露给任何 MCP 客户端。
注意：stdio 模式下 **stdout 是协议通道**，调试信息只能写 stderr！

单独运行（等待客户端连接，不会自己退出）：
  uv run python stage2/mcp_server.py
"""

from __future__ import annotations

from pathlib import Path

from mcp.server import MCPServer

mcp = MCPServer("stage2-notes")
NOTES = Path(__file__).resolve().parent / "workspace" / "notes"


@mcp.tool()
def word_count(text: str) -> str:
    """统计一段文本的字符数、词数和行数。"""
    return (
        f"字符数 {len(text)}，词数 {len(text.split())}，"
        f"行数 {len(text.splitlines())}"
    )


@mcp.tool()
def read_note(name: str) -> str:
    """读取 notes 目录下的一篇笔记，例如 meeting.md。"""
    target = (NOTES / name).resolve()
    if not target.is_relative_to(NOTES):
        return f"错误：路径越界：{name!r}"
    if not target.is_file():
        return f"错误：笔记不存在：{name!r}。可以先用 list_notes 看看。"
    return target.read_text(encoding="utf-8")


@mcp.tool()
def list_notes() -> str:
    """列出 notes 目录里的所有笔记文件名。"""
    return "\n".join(sorted(p.name for p in NOTES.glob("*.md"))) or "（空目录）"


@mcp.resource("note://{name}")
def note_resource(name: str) -> str:
    """把一篇笔记暴露成 MCP resource，供客户端直接读取。"""
    return read_note(name)


@mcp.prompt()
def summarize_note(name: str) -> str:
    """生成一个“总结某篇笔记”的提示词模板。"""
    return f"请阅读笔记 {name}，用三点总结，并引用原文中的依据。"


if __name__ == "__main__":
    mcp.run()  # 默认 stdio；也可以 mcp.run(transport="sse")
