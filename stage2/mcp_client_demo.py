"""阶段 2 · 自己写一个 MCP client，连上 mcp_server.py。

运行：uv run python stage2/mcp_client_demo.py

它会做四件事：
  1. 用 stdio 启动 server
  2. initialize + list_tools + call_tool
  3. list_resources + read_resource
  4. list_prompts + get_prompt

不需要 Claude Desktop，也不需要任何额外交互。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp.client import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

STAGE2 = Path(__file__).resolve().parent
ROOT = STAGE2.parent
SERVER = STAGE2 / "mcp_server.py"


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        cwd=str(ROOT),
        env={"PYTHONUTF8": "1"},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=" * 60)
            tools = await session.list_tools()
            print("工具：", [tool.name for tool in tools.tools])
            for tool in tools.tools:
                print(f"  · {tool.name}: {tool.description}")

            print("=" * 60)
            result = await session.call_tool("word_count", {"text": "hello 世界"})
            print("call_tool(word_count) 返回：")
            for block in result.content:
                print(" ", block)

            result = await session.call_tool("read_note", {"name": "meeting.md"})
            print("call_tool(read_note) 返回：")
            for block in result.content:
                text = getattr(block, "text", str(block))
                print(" ", text[:300])

            print("=" * 60)
            resources = await session.list_resources()
            print("静态 resources：", [str(item.uri) for item in resources.resources])
            templates = await session.list_resource_templates()
            print(
                "resource templates：",
                [item.uri_template for item in templates.resource_templates],
            )
            resource = await session.read_resource("note://meeting.md")
            print("read_resource(note://meeting.md) 返回：")
            for block in resource.contents:
                text = getattr(block, "text", str(block))
                print(" ", text[:200])

            print("=" * 60)
            prompts = await session.list_prompts()
            print("prompts：", [item.name for item in prompts.prompts])
            if prompts.prompts:
                prompt = await session.get_prompt(
                    "summarize_note", {"name": "meeting.md"}
                )
                print("get_prompt 返回：")
                for message in prompt.messages:
                    print(" ", message.content)


if __name__ == "__main__":
    asyncio.run(main())
