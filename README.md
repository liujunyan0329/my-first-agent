# my-first-agent

AI Agent 学习项目的 uv 工程根目录。`.venv`、`.env`、`pyproject.toml` 由两个阶段共用。

```
my-first-agent\
├── stage0\        ← 阶段 0：API 调用、多轮对话、流式、async、上下文边界
├── stage1\        ← 阶段 1：工具调用、核心循环、300 行 agent.py
├── src\           ← uv 初始化时生成的包（暂未使用）
├── .env           ← API key（不要提交）
├── .venv\         ← 共享虚拟环境
└── pyproject.toml
```

## 阶段代码

| 阶段 | 目录 | 内容 | 对应教程 |
|---|---|---|---|
| 阶段 0 | [`stage0/`](stage0/) | API 调用、多轮对话、流式、async、上下文边界 | 课程资料《阶段0-动手任务教程》 |
| 阶段 1 | [`stage1/`](stage1/) | 工具调用、核心循环、300 行 `agent.py` | 课程资料《阶段1-动手任务教程》 |

## 运行方式

需要 Python 3.13+、[uv](https://docs.astral.sh/uv/)，以及一个 DeepSeek API key。

```bash
git clone git@github.com:liujunyan0329/my-first-agent.git
cd my-first-agent
uv sync
cp .env.example .env    # 然后编辑 .env，填入 DEEPSEEK_API_KEY
```

**在项目根目录运行**（这样才能找到 `.venv` 和 `.env`）：

```bash
PYTHONUTF8=1 uv run python stage0/hello_llm.py
PYTHONUTF8=1 uv run python stage1/agent.py "北京现在几点？比东京早还是晚几小时？把结论写进 result.txt"
```

> Windows 下带 `PYTHONUTF8=1`，避免中文编码问题。
