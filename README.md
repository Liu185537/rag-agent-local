# RAG-Agent Local（面试展示版）
[![CI](https://github.com/Liu185537/rag-agent-local/actions/workflows/ci.yml/badge.svg)](https://github.com/Liu185537/rag-agent-local/actions/workflows/ci.yml)

这是一个面向面试演示的本地优先 `RAG + Agent` 项目。
项目已主动移除公司专属的 IP、端口和 API Key 配置，默认可在个人电脑本地运行。

## 文档导航（新手优先）

- 架构总览与时序图： [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 面试演示脚本与讲解顺序： [docs/INTERVIEW_PLAYBOOK.md](docs/INTERVIEW_PLAYBOOK.md)

如果你是第一次接触这个项目，建议先看架构文档，再回到 README 跑命令。

## 1. 项目展示能力

- 端到端 RAG 流程：
  - 数据摄取 -> 分块 -> 向量化 -> 向量索引
  - 混合检索（向量检索 + BM25 + RRF 融合）
  - 轻量重排（语义 + 词法 + 覆盖度）
  - 带引用标注的答案生成
- Agent 工作流：
  - Planner -> Tool 执行 -> 回复合成
  - 会话记忆（SQLite）
  - 用户画像记忆工具（读取/更新 profile）
- 工程化基础：
  - `.env` 配置管理
  - SQLite + Chroma 本地持久化
  - 支持 txt/md/csv/json/pdf/docx/xlsx 上传解析
  - 请求指标采集与轻量可视化面板
  - 快速评测脚本（可复现指标）

## 2. 技术栈

- 后端：FastAPI
- 向量库：Chroma（本地持久化）
- 记忆与元数据：SQLite
- LLM/Embedding：支持 Ollama（本地）与 SiliconFlow（云端 API），并提供不可用时的降级逻辑

## 3. 快速开始

### 3.1 前置条件

- Python 3.10+
- 二选一：
  - 本地 Ollama（`http://127.0.0.1:11434`）
  - SiliconFlow API Key（云端模型）

### 3.2 安装

```bash
cd Rag-Agent
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

### 3.3 配置

```bash
copy .env.example .env
```

默认示例配置使用 SiliconFlow。请至少确认下列字段：

```env
RAG_AGENT_LLM_PROVIDER=siliconflow
RAG_AGENT_SILICONFLOW_API_KEY=YOUR_SILICONFLOW_API_KEY
RAG_AGENT_SILICONFLOW_CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
RAG_AGENT_SILICONFLOW_EMBED_MODEL=BAAI/bge-m3
```

如果你要切回本地 Ollama：

```env
RAG_AGENT_LLM_PROVIDER=ollama
RAG_AGENT_OLLAMA_BASE_URL=http://127.0.0.1:11434
RAG_AGENT_OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct
RAG_AGENT_OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 3.4 启动 API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8008 --reload
```

Swagger 文档：`http://127.0.0.1:8008/docs`

## 4. 演示流程

### 4.1 导入演示知识库

```bash
python scripts/ingest_demo.py --base-url http://127.0.0.1:8008 --namespace demo
```

### 4.2 发起提问

```bash
curl -X POST "http://127.0.0.1:8008/api/v1/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"namespace\":\"demo\",\"message\":\"How should I handle price objections in first chat?\"}"
```

### 4.3 直接上传文件

```bash
curl -X POST "http://127.0.0.1:8008/api/v1/knowledge/upload" ^
  -F "namespace=demo" ^
  -F "file=@data\\sample_knowledge.md"
```

### 4.4 查看指标与可视化

- 指标 JSON：`http://127.0.0.1:8008/api/v1/metrics`
- Dashboard：`http://127.0.0.1:8008/dashboard`
- Playground：`http://127.0.0.1:8008/playground`
  - 页面布局参考 `Knowledge.vue` 交互风格（左侧知识区 + 右侧聊天/配置 Tab）。
  - Agent 配置改为后端持久化（`/api/v1/agent-config`），不再仅保存在浏览器本地。

### 4.5 运行轻量评测

```bash
python scripts/run_eval.py --base-url http://127.0.0.1:8008 --namespace demo
```

脚本会输出：
- 控制台摘要（命中率 + 延迟 p50/p90/p95）
- `eval/reports/*.json` 详细报告
- `eval/reports/*.csv` 样本级结果

### 4.6 生成更大评测集（50+）

```bash
python scripts/generate_eval_dataset.py --size 60 --output eval/dataset_v2.jsonl
python scripts/run_eval.py --dataset eval/dataset_v2.jsonl --namespace demo
```

### 4.7 面试展示一键流程

```bash
python scripts/readiness_check.py --base-url http://127.0.0.1:8008 --namespace demo
python scripts/interview_demo.py --base-url http://127.0.0.1:8008 --namespace interview_demo
```

PowerShell 辅助脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/interview_demo.ps1
```

## 5. 项目结构

```text
Rag-Agent/
  app/
    main.py
    api/schemas.py
    core/{config.py,database.py,logging.py,observability.py}
    llm/client.py
    rag/{chunker.py,document_parser.py,embedding.py,indexer.py,retriever.py,reranker.py}
    agent/{tools.py,orchestrator.py}
  scripts/{ingest_demo.py,run_eval.py,generate_eval_dataset.py,readiness_check.py,interview_demo.py}
  tests/
  docs/{ARCHITECTURE.md,INTERVIEW_PLAYBOOK.md}
  eval/dataset.jsonl
  eval/dataset_v2.jsonl
  docker-compose.yml
  Dockerfile
  data/sample_knowledge.md
  .env.example
  pyproject.toml
```

## 6. 代码导读路径（按 30-60 分钟设计）

### 第一阶段：先跑通主流程（10 分钟）

1. 看入口接口如何串联： [app/main.py](app/main.py)
2. 运行一次知识入库： [scripts/ingest_demo.py](scripts/ingest_demo.py)
3. 发起一次聊天请求，观察响应里的 citations 和 trace

### 第二阶段：理解 Agent 怎么做决策（15 分钟）

1. 看编排器： [app/agent/orchestrator.py](app/agent/orchestrator.py)
2. 看工具层： [app/agent/tools.py](app/agent/tools.py)
3. 对照架构时序： [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### 第三阶段：理解检索与重排（20 分钟）

1. 混合检索主逻辑： [app/rag/retriever.py](app/rag/retriever.py)
2. 向量索引与查询： [app/rag/indexer.py](app/rag/indexer.py)
3. 嵌入与回退机制： [app/rag/embedding.py](app/rag/embedding.py)
4. 重排信号： [app/rag/reranker.py](app/rag/reranker.py)

### 第四阶段：理解存储与可观测（10 分钟）

1. 数据表与查询： [app/core/database.py](app/core/database.py)
2. 配置来源： [app/core/config.py](app/core/config.py)
3. 指标采集： [app/core/observability.py](app/core/observability.py)

### 第五阶段：理解工程化与验证（可选）

1. API 烟雾测试： [tests/test_api_smoke.py](tests/test_api_smoke.py)
2. 测试夹具与环境隔离： [tests/conftest.py](tests/conftest.py)
3. 自动化评测脚本： [scripts/run_eval.py](scripts/run_eval.py)

## 7. 面试讲解要点

- 为什么在真实业务语料下，混合检索通常优于纯向量检索。
- 相比单轮 Prompt，为什么 Planner + Tool 的 Agent 方案更可控。
- 如何量化效果：
  - retrieval hit@k
  - citation coverage
  - 任务成功率 / 延迟 / 成本
- 下一步优化方向：
  - 接入更强 reranker 模型
  - 加入异步摄取流水线
  - 增强 Guardrails 与可观测性（例如 tracing）

## 8. 安全与隐私

- 默认配置不包含公司内网或专有端点。
- 代码仓库不提交 API Key。
- `.env.example` 默认仅本地可运行。

## 9. 测试与 Docker

运行测试：

```bash
pip install -r requirements-dev.txt
pytest -q
```

CI：
- GitHub Actions 工作流：`.github/workflows/ci.yml`
- 流水线阶段：
  - `ruff check app tests scripts`
  - `pytest -q tests`
  - `docker build`

使用 Docker 启动：

```bash
docker compose up --build
```

启动后访问：
- `http://127.0.0.1:8008/docs`
- `http://127.0.0.1:8008/playground`
