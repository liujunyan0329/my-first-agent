# 阶段 4 · 编排与多智能体

对应教程：`E:\ai agent\docs\阶段4-动手任务教程.md`
对应路线：`01-学习路线.md` → 阶段 4 → 动手任务 6 条

## 文件

| 文件 | 对应任务 | 说明 |
|---|---|---|
| `tools.py` | 全部 | 复用阶段 2 注册机制的 6 个工具（含 0.8s 延迟的 mock_weather） |
| `graph_agent.py` | ① | 用 LangGraph 重写阶段 1 的 agent，支持并行工具节点 |
| `compare_loop.md` | ① | 手写循环 vs 图：可读性、可控性、调试、扩展性对比 |
| `parallel_tools.py` | ② | 同一批 tool_calls 串行 vs `asyncio.gather` 的延迟实测 |
| `checkpoint_demo.py` | ③ | SQLite 检查点：`os._exit(7)` 后从断点恢复 |
| `hitl_demo.py` | ④ | `interrupt()` + `Command(resume=...)`：跨进程人工审批 |
| `code_tasks.py` | ⑤⑥ | 6 个编码任务 + 隐藏测试子进程判分器 |
| `generator_critic.py` | ⑤ | 单 agent vs 生成者—检查者双 agent |
| `cost_experiment.py` | ⑥ | 成功率、调用次数、token、费用、耗时对比 |

## 运行

在项目根目录：

```bash
uv sync

# ① 图 agent
uv run python stage4/graph_agent.py "北京现在几点？比东京早还是晚几小时？"
uv run python stage4/graph_agent.py --parallel "查北京、东京、纽约的天气"
uv run python stage4/graph_agent.py --graph

# ② 并行工具
uv run python stage4/parallel_tools.py

# ③ 断点恢复（第一次会以 exit code 7 退出，这是故意的）
uv run python stage4/checkpoint_demo.py --crash
uv run python stage4/checkpoint_demo.py --resume
uv run python stage4/checkpoint_demo.py --history

# ④ 人工审批
uv run python stage4/hitl_demo.py run "写一份本周学习计划"
uv run python stage4/hitl_demo.py status
uv run python stage4/hitl_demo.py approve     # 或 reject

# ⑤ 单题对比
uv run python stage4/generator_critic.py --task eval_expr

# ⑥ 成本实验（6 题 × 3 次，约 1.5 分钟）
uv run python stage4/cost_experiment.py --repeat 3

# 类型检查
uv run pyright stage4
```

## 产物（git 忽略）

- `stage4/runtime/`：SQLite 检查点、执行日志、实验 JSON
- `stage4/workspace/`：agent 写入的文件（报告、备忘录）
