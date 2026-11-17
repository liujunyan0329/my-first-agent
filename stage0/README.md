# 阶段 0 · 基本功代码

对应课程资料中的《阶段0-动手任务教程》（教程文档未随仓库发布）。

| 文件 | 任务 | 内容 |
|---|---|---|
| `hello_llm.py` | ② | 第一次 LLM API 调用，打印回复和 token |
| `chat_loop.py` | ③ | 多轮对话，手动维护 `messages` 列表 |
| `streaming.py` | ④ | 同步流式输出 + 首字延迟（TTFT） |
| `async_demo.py` | ④ | `async/await`、`asyncio.gather`、阻塞事件循环实验 |
| `overflow.py` | ⑤ | 超 `max_tokens` / 超 context window 的报错实验 |
| `mini_chat.py` | 结业小测 | 异步流式 + 滑动窗口 + token 统计 |

## 运行

`.venv`、`.env`、`pyproject.toml` 都在上一级项目根目录，所以**在项目根目录运行**：

```bash
# 先把 .env.example 复制为 .env 并填入 DEEPSEEK_API_KEY
PYTHONUTF8=1 uv run python stage0/hello_llm.py
PYTHONUTF8=1 uv run python stage0/chat_loop.py
PYTHONUTF8=1 uv run python stage0/streaming.py
PYTHONUTF8=1 uv run python stage0/async_demo.py
PYTHONUTF8=1 uv run python stage0/overflow.py
PYTHONUTF8=1 uv run python stage0/mini_chat.py
```

> Windows 下务必带 `PYTHONUTF8=1`，否则可能撞上中文编码错误（见教程开头）。
