# 阶段 5 · 工程化：可观测、可评估、可负担

对应教程：`E:\ai agent\docs\阶段5-动手任务教程.md`
对应路线：`01-学习路线.md` → 阶段 5 → 动手任务 6 条

阶段 5 不引入新框架，做的是把前四个阶段的 demo 变成"能交付"的系统：
**trace 让你看见，eval 让你度量，caching / routing 让你付得起。**

## 架构

```
                     TracedAgent
                  agent.run
                   ├─ llm.call #1 ──→ Tracer ──→ trace.jsonl / trace_otel.json
                   ├─ tool.xxx                   终端瀑布图
                   └─ llm.call #2

  eval_set.jsonl（30 题）
        │
        ▼
  Retriever（vector / hybrid）──→ deepseek-chat ──→ keyword + judge
        │                                              │
        └──────────────────────────────────────────────┘
                              ▼
                   通过率 + 失败模式（retrieval_miss / generation_miss / judge_fail）

   index_eval（切块 / top-k / 检索方式）
   caching_demo（前缀缓存）
   routing_demo（模型路由）
        └──────────→ cost_report.md → 本 README
```

## 文件

| 文件 | 对应任务 | 说明 |
|---|---|---|
| `tracing.py` | ① | 最小 trace：span 树、token/费用、JSONL、瀑布图、OTel 导出 |
| `traced_agent.py` | ① | 带完整 trace 的 agent，含错误 span 演示 |
| `eval_set.jsonl` | ② | 30 条课程问答，覆盖 5 个实验，带参考要点和关键词组 |
| `rag_eval.py` | ②③ | 双信号判分、失败模式分类、成本记录 |
| `retrieval_v2.py` | ③ | 纯向量 / BM25+向量混合检索，标题注入开关 |
| `index_eval.py` | ③ | 6 个配置的消融实验 |
| `caching_demo.py` | ④ | prompt caching 命中率、费用、延迟实测 |
| `routing_demo.py` | ⑤ | always_chat / always_premium / routed 三策略对比 |
| `cost_report.py` | ⑥ | 汇总所有实验，生成 `runtime/cost_report.md` |

## 运行

在项目根目录（`E:/ai agent/code/my-first-agent`）：

```bash
uv sync

# ① trace
PYTHONUTF8=1 uv run python stage5/traced_agent.py "北京现在几点？顺便查一下北京的天气"
PYTHONUTF8=1 uv run python stage5/traced_agent.py "阿特兰蒂斯现在几点？查不到就说查不到"
PYTHONUTF8=1 uv run python stage5/traced_agent.py --otel --no-render "查上海和东京的天气"

# ② 30 题基线（约 1 分钟，¥0.06）
PYTHONUTF8=1 uv run python stage5/rag_eval.py --label baseline

# ③ 消融实验（6 配置 × 30 题；首次会建本地索引，约 5 分钟，¥0.6）
PYTHONUTF8=1 uv run python stage5/index_eval.py

# ④ prompt caching（约 10 秒）
PYTHONUTF8=1 uv run python stage5/caching_demo.py

# ⑤ 模型路由（3 策略 × 30 题，reasoner 较慢，约 5 分钟，¥0.5）
PYTHONUTF8=1 uv run python stage5/routing_demo.py

# ⑥ 汇总报告
PYTHONUTF8=1 uv run python stage5/cost_report.py

# 类型检查
uv run pyright stage5
```

## 实测结果（2026-11-23，30 题）

### 检索消融

| 配置 | 块数 | top-k | 方式 | 双通过 | Δ | 费用 |
|---|---|---|---|---|---|---|
| `vector500_k4`（基线） | 569 | 4 | vector | 20/30 (67%) | — | ¥0.061 |
| `heading500_k4` | 569 | 4 | vector | 18/30 (60%) | -2 | ¥0.091 |
| `vector500_k8` | 569 | 8 | vector | 22/30 (73%) | +2 | ¥0.112 |
| `hybrid500_k6` | 569 | 6 | hybrid | 29/30 (97%) | **+9** | ¥0.116 |
| `hybrid800_k6` | 423 | 6 | hybrid | 28/30 (93%) | +8 | ¥0.163 |
| `hybrid300_k6` | 864 | 6 | hybrid | 26/30 (87%) | +6 | ¥0.113 |

结论：**主要瓶颈是检索可达性，不是生成**。混合检索把 `retrieval_miss`
从 6 条清到 0；只加 top-k 或只改切块，收益小得多。
详见 `runtime/cost_report.md`。

> 同一配置多次重跑会在 20~22/30 之间波动（本机实测 20、21、22 都出现过）。
> 表里的基线 20/30 是消融实验那一次；`cost_report.md` 的第 1 节可能显示另一次
> 重跑的 22/30。这是 eval 噪音，不是数据错误。

### prompt caching

- 固定前缀 2544 字，约 1444 token；
- 第 2 次起命中 1280 token（89%）；
- 5 次调用费用 ¥0.0057，不缓存 ¥0.0149，节省 62%；
- 每次改前缀开头 → 命中率 0%。

### 模型路由

| 策略 | 双通过 | 生成费用 | 模型构成 |
|---|---|---|---|
| `always_chat` | 22/30 | ¥0.0339 | chat × 30 |
| `always_premium` | 22/30 | ¥0.1937 | reasoner × 30 |
| `routed` | 23/30 | ¥0.0677 | chat × 22 + reasoner × 8 |

结论：在这套课程问答上，reasoner 没有提升质量（22 vs 22），成本是 chat 的 5.7 倍。
路由相对 premium 省 65%，但仍比 chat 贵 2 倍。**默认选 always_chat**；
只有当 eval 证明贵模型在某类难题上显著更好时，才值得路由。

## 产物（git 忽略）

- `runtime/`：trace、eval 结果、caching/routing JSON、`cost_report.md`
- `index_500h/`、`index_300/`、`index_800/`：消融用的本地向量索引（换机器重建）

## 已知限制

- eval 只有 30 条，覆盖 5 个实验，样本小；±1~2 题是噪音；
- judge 与被测模型同源，有自偏好；关键词规则会漏同义表达；
- 切块/检索参数是在这 30 题上调的，存在过拟合风险；严肃项目要留 hold-out；
- 价格常量按本账户 2026-11 价目表设置，换模型/账户需改常量；
- prompt caching 命中率由服务端决定，不是常数；
- 路由规则是手工启发式，没有独立验证集调参。
