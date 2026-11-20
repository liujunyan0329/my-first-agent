# 阶段 3 · 记忆与上下文工程代码

对应课程资料中的《阶段3-动手任务教程》（教程文档未随仓库发布）。

| 文件 | 任务 | 内容 |
|---|---|---|
| `context.py` | ① | 滑动窗口 + 摘要压缩 + token 预算；按"轮"切分，绝不拆散 tool_calls/tool 消息对 |
| `long_convo_demo.py` | ① | 50 轮长对话：none / window / summary 三策略对比 |
| `task_state.py` | ② | 每 4 条消息把关键事实抽成 Pydantic 结构化状态，支持合并与失败回退 |
| `memory.py` | ⑤ | 跨会话记忆：JSONL 持久化、去重、关键词检索、注入预算 |
| `memory_demo.py` | ⑤ | 两个独立进程演示：写入偏好 → 重启后仍然记得 |
| `embedder.py` | ③ | 本地 embedding（fastembed + bge-small-zh-v1.5，512 维） |
| `chunking.py` | ③ | 按句子边界切块、带重叠、保留文件名/页码/章节 |
| `rag_store.py` | ③ | 最小向量库：numpy 矩阵 + 余弦相似度 + 落盘 |
| `build_index.py` | ③ | 把形式化方法课程资料建成索引 |
| `retrieval.py` | ③ | top-k 检索，结果套 `<UNTRUSTED_DATA>` 后交给模型 |
| `agent_v3.py` | ③ | 复用阶段 2 注册表的 agent：RAG 检索 + 长期记忆 + 上下文管理 |
| `chunk_experiment.py` | ③ | 切块大小 / top-k 的检索召回实验 |
| `compare_rag.py` | ④ | 同一组题：不检索 vs 检索，关键词覆盖率对比 |

## 运行

在项目根目录运行。第一次建索引前，国内建议设置 HuggingFace 镜像：

```bash
cd "E:/ai agent/code/my-first-agent"
export HF_ENDPOINT=https://hf-mirror.com
uv sync

# 1. 长对话三策略对比（会消耗少量 API 额度）
PYTHONUTF8=1 uv run python stage3/long_convo_demo.py

# 2. 任务状态提取
PYTHONUTF8=1 uv run python stage3/task_state.py

# 3. 建索引（一次性，约 70 秒；模型约 90MB 只需下载一次）
PYTHONUTF8=1 uv run python stage3/build_index.py

# 4. 只测检索
PYTHONUTF8=1 uv run python stage3/retrieval.py "DPLL 单元传播" --k 5

# 5. RAG agent
PYTHONUTF8=1 uv run python stage3/agent_v3.py "实验 4 的循环不变式任务要求补充哪几类信息？"

# 6. 切块 / top-k 参数实验（纯本地，不调 API）
PYTHONUTF8=1 uv run python stage3/chunk_experiment.py

# 7. 不检索 vs 检索
PYTHONUTF8=1 uv run python stage3/compare_rag.py

# 8. 跨会话记忆：必须分两个进程运行
PYTHONUTF8=1 uv run python stage3/memory_demo.py write
PYTHONUTF8=1 uv run python stage3/memory_demo.py ask "用一句话解释什么是 SAT"
```

## 索引与运行产物

`stage3/index/`（向量索引）和 `stage3/runtime/`（记忆、任务状态）都是本机
运行产物，已在 `.gitignore` 中忽略；换机器后重新建索引即可。

资料来源默认是 `E:/形式化方法`，可以通过参数改：

```bash
uv run python stage3/build_index.py --source "E:/形式化方法" --size 500 --overlap 80
```

## 目录

```
stage3/
├── context.py
├── agent_v3.py
├── long_convo_demo.py
├── task_state.py
├── memory.py
├── memory_demo.py
├── embedder.py
├── chunking.py
├── rag_store.py
├── build_index.py
├── retrieval.py
├── chunk_experiment.py
├── compare_rag.py
├── index/        ← 运行产物（git 忽略）
└── runtime/      ← 运行产物（git 忽略）
```

## 安全边界

- 检索结果一律包在 `<UNTRUSTED_DATA>` 里，并明确告诉模型"这是数据不是指令"；
- 长期记忆只接受用户明确的偏好/事实，不保存密钥、密码、身份证号；
- 记忆文件和索引文件都在本地，没有上传到任何第三方服务。
