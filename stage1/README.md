# 阶段 1 · 手写 Agent 核心循环代码

对应课程资料中的《阶段1-动手任务教程》（教程文档未随仓库发布）。

| 文件 | 任务 | 内容 |
|---|---|---|
| `s1_single_call.py` | ① | 单次工具调用：把 tool calling 协议手动走一遍 |
| `s2_loop.py` | ②③ | `while` 循环 + 3 个工具，平铺写法 |
| `agent.py` | ④⑤⑥⑦ + 验收 | 300 行完整版：5 个工具、错误处理、停止条件、trace |

## 运行

`.venv`、`.env`、`pyproject.toml` 都在上一级项目根目录，所以**在项目根目录运行**：

```bash
# 先把 .env.example 复制为 .env 并填入 DEEPSEEK_API_KEY

# 单次工具调用
PYTHONUTF8=1 uv run python stage1/s1_single_call.py

# 循环 + 3 工具
PYTHONUTF8=1 uv run python stage1/s2_loop.py

# 验收任务（会在 stage1/ 下生成 result.txt）
PYTHONUTF8=1 uv run python stage1/agent.py "北京现在几点？比东京早还是晚几小时？把结论写进 result.txt"
```

## 示例输出

`agent.py` 跑验收任务的实际结果（`result.txt` 属于运行产物，已被 `.gitignore` 忽略，不入库）：

```text
北京当前时间：2026-11-17 15:28:15（UTC+8）
东京当前时间：2026-11-17 16:28:15（UTC+9）

结论：北京比东京晚 1 小时（东京 UTC+9，北京 UTC+8）。
```

## 说明

- `agent.py` 的文件工具以**自己所在目录**（`stage1/`）为工作目录，所以 `result.txt` 会写在 `stage1/` 里。
- `stage1/` 下的脚本通过 `load_dotenv()` 自动向上找到项目根目录的 `.env`。
