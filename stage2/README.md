# 阶段 2 · 工具与协议代码

对应课程资料中的《阶段2-动手任务教程》（教程文档未随仓库发布）。

| 文件 | 任务 | 内容 |
|---|---|---|
| `registry.py` | ① | `@tool` 装饰器 + Pydantic 自动生成 JSON Schema + 参数校验 |
| `builtin_tools.py` | ② | 6 个注册式工具：时间、算数、文件读写、列目录、代码执行 |
| `agent_v2.py` | ② | 用注册表重写的 agent，危险工具需要审批 |
| `sandbox.py` | ④ | 本地受限子进程 + Docker 两种代码执行后端 |
| `mcp_server.py` | ③ | 自写 MCP server：3 tools + 1 resource template + 1 prompt |
| `mcp_client_demo.py` | ③ | 自己写 MCP client，通过 stdio 连接并调用 |
| `mcp_agent.py` | ③ | 让 agent 通过 MCP 协议使用工具，不 import 任何工具函数 |
| `injection_demo.py` | ⑤ | 间接 prompt injection：朴素 RAG 可复现，加固版可挡住 |

## 运行

在项目根目录运行：

```bash
cd "E:/ai agent/code/my-first-agent"

# 1. 看自动生成的工具 schema + Pydantic 参数校验
PYTHONUTF8=1 uv run python stage2/builtin_tools.py

# 2. 本地沙箱（临时目录 + 超时 + 清洗环境变量）
PYTHONUTF8=1 uv run python stage2/sandbox.py

# 3. agent_v2：6 个注册式工具
PYTHONUTF8=1 uv run python stage2/agent_v2.py "北京现在几点？写进 time.txt"

# 4. 代码执行 + 文件写入（危险工具需要 --allow-dangerous）
PYTHONUTF8=1 uv run python stage2/agent_v2.py --allow-dangerous \
  "用 Python 算 1 到 100 的和，把结果写进 sum.txt"

# 5. MCP server + 自己的 client
PYTHONUTF8=1 uv run python stage2/mcp_client_demo.py

# 6. 让 agent 通过 MCP 使用工具
PYTHONUTF8=1 uv run python stage2/mcp_agent.py "统计 'hello 世界' 的字符数"

# 7. prompt injection 实验
PYTHONUTF8=1 uv run python stage2/injection_demo.py
```

## Docker 沙箱（可选，但验收建议跑）

先启动 Docker Desktop，然后准备镜像。国内直连 Docker Hub 可能失败，可以用镜像站：

```bash
docker pull docker.m.daocloud.io/library/python:3.13-alpine
docker tag docker.m.daocloud.io/library/python:3.13-alpine python:3.13-alpine
```

再运行：

```bash
SANDBOX_BACKEND=docker PYTHONUTF8=1 uv run python stage2/sandbox.py
SANDBOX_BACKEND=docker PYTHONUTF8=1 uv run python stage2/agent_v2.py --allow-dangerous \
  "用 Python 算 1 到 100 的和"
```

Docker 后端会加：`--network none`、`--read-only`、`--memory 256m`、`--cpus 0.5`、
`--pids-limit 64`、`--cap-drop ALL`、`--security-opt no-new-privileges`。

## 接入 Claude Desktop

编辑 `%APPDATA%\Claude\claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "stage2-notes": {
      "command": "E:/ai agent/code/my-first-agent/.venv/Scripts/python.exe",
      "args": ["E:/ai agent/code/my-first-agent/stage2/mcp_server.py"]
    }
  }
}
```

重启 Claude Desktop 后，它应该能看到 `word_count`、`read_note`、`list_notes` 三个工具，
以及 `note://{name}` resource 和 `summarize_note` prompt。

> stdio 模式下 **stdout 是 JSON-RPC 协议通道**，调试信息一律写 stderr，
> 否则客户端会解析失败。

## 目录

```
stage2/
├── registry.py
├── builtin_tools.py
├── agent_v2.py
├── sandbox.py
├── mcp_server.py
├── mcp_client_demo.py
├── injection_demo.py
└── workspace/          ← 文件工具的工作目录
    └── notes/
        ├── meeting.md
        ├── poisoned.md
        └── poisoned_rag.md
```

生成的 `workspace/*.txt` 属于运行产物，已在 `.gitignore` 中忽略。
