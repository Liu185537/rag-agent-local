# 面试演示手册（逐分钟讲解版）

## 0. 演示目标（先说清楚）

在 8-10 分钟内证明三件事：

1. 这不是单纯 Prompt 拼接，而是完整的 RAG + Agent 系统。
2. 项目具备工程化能力（可运行、可测试、可评测、可复现）。
3. 你不仅会用，还能解释设计取舍与后续优化方向。

参考文档：

- 架构总览：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 快速命令入口：[README.md](README.md)

## 1. 预备清单（开始前 1 分钟）

确保以下条件满足：

1. 服务可启动，访问 http://127.0.0.1:8008/docs 正常。
2. Playground 可访问，地址 http://127.0.0.1:8008/playground。
3. 演示文件存在：[data/sample_knowledge.md](data/sample_knowledge.md)。
4. 演示脚本存在：[scripts/interview_demo.py](scripts/interview_demo.py)。

建议提前执行：

1. uvicorn app.main:app --host 127.0.0.1 --port 8008 --reload
2. python scripts/readiness_check.py --base-url http://127.0.0.1:8008 --namespace demo

## 2. 逐分钟演示脚本（8-10 分钟）

### 第 0-1 分钟：开场定位

你可以这样说：

1. 这是本地优先的 RAG-Agent，支持知识入库、混合检索、引用回答与会话记忆。
2. 我会现场走一遍从入库到问答再到报告的完整流程。

### 第 1-2 分钟：打开 Playground，展示入口能力

操作：

1. 打开 http://127.0.0.1:8008/playground。
2. 指出页面包含知识上传、聊天、配置等核心入口。

讲解重点：

1. 前端只是操作面板，核心能力都在后端 API。
2. 所有状态可以回溯到数据库和指标接口。

### 第 2-3 分钟：上传知识并解释入库链路

操作：

1. 上传 [data/sample_knowledge.md](data/sample_knowledge.md)。
2. 展示返回的入库结果（文档数、分块数）。

讲解重点：

1. 入库流程是 文档解析 -> 分块 -> 向量化 -> 写入 Chroma + SQLite。
2. 其中 SQLite 中的 chunks 也会用于 BM25 检索。

代码对应：

- 入库 API：[app/main.py](app/main.py)
- 分块逻辑：[app/rag/chunker.py](app/rag/chunker.py)
- 向量与索引：[app/rag/embedding.py](app/rag/embedding.py), [app/rag/indexer.py](app/rag/indexer.py)

### 第 3-5 分钟：连续问答，展示引用与检索效果

操作：

1. 提一个异议处理问题。
2. 再提一个跟进节奏问题。
3. 展示回答中的引用信息和会话连续性。

讲解重点：

1. 检索是混合策略：向量 + BM25 + RRF + 重排。
2. 回答附带 citations 与 trace，便于可解释和调试。

代码对应：

- 检索主逻辑：[app/rag/retriever.py](app/rag/retriever.py)
- 重排逻辑：[app/rag/reranker.py](app/rag/reranker.py)
- Agent 编排：[app/agent/orchestrator.py](app/agent/orchestrator.py)

### 第 5-6 分钟：修改 Agent 配置并验证持久化

操作：

1. 保存一个新的 Agent 配置（名称、描述、指令）。
2. 刷新页面，确认配置仍存在。

讲解重点：

1. 配置不是前端本地缓存，而是后端持久化。
2. 这体现了多会话、多 namespace 的工程可用性。

代码对应：

- 配置接口：[app/main.py](app/main.py)
- 持久化实现：[app/core/database.py](app/core/database.py)

### 第 6-7 分钟：打开 Dashboard，展示可观测性

操作：

1. 打开 http://127.0.0.1:8008/dashboard。
2. 展示总请求、错误率、平均延迟、最近会话。

讲解重点：

1. 不是黑盒对话系统，接口级指标可见。
2. 出问题时可基于 request id、trace、metrics 排查。

代码对应：

- 指标采集：[app/core/observability.py](app/core/observability.py)

### 第 7-9 分钟：运行一键演示脚本并展示报告

操作：

1. 执行命令：
   python scripts/interview_demo.py --base-url http://127.0.0.1:8008 --namespace interview_demo
2. 展示生成的两份报告。

报告路径：

- eval/reports/interview_demo_*.json
- eval/reports/interview_demo_*.md

讲解重点：

1. 演示流程可脚本化复现。
2. 能沉淀证据，便于面试后复盘与二次沟通。

### 第 9-10 分钟：总结与扩展

你可以这样收尾：

1. 检索质量可进一步接入更强 reranker。
2. 入库链路可异步化提升吞吐。
3. 观测层可加入 tracing 与告警策略。

## 3. 高频追问与建议回答

### Q1：为什么不用纯向量检索？

建议回答：

1. 纯向量在关键词强约束场景容易漏召回。
2. BM25 能补词法精确匹配，RRF 融合后更稳。

### Q2：Agent 的价值是什么？

建议回答：

1. 把规划、工具调用、回复分层，流程更可控。
2. 能把画像、检索和生成统一到可追踪链路里。

### Q3：如果本地模型挂了怎么办？

建议回答：

1. 向量侧有回退 embedding，服务不断。
2. 聊天侧有降级回答，接口不崩，体验可预期。

## 4. 演示失败应急预案（很实用）

### 情况 A：服务起不来

1. 先看健康检查接口 /api/v1/health。
2. 再看配置文件与本地路径权限。

### 情况 B：回答无引用或引用很少

1. 先确认是否已成功入库。
2. 再调用 /api/v1/retrieve 看召回结果。

### 情况 C：回答质量波动

1. 检查模型是否可用（Ollama 是否启动）。
2. 检查 namespace 是否正确、知识是否混入无关内容。

## 5. 一页命令清单

1. 启动服务
   uvicorn app.main:app --host 127.0.0.1 --port 8008 --reload

2. 就绪检查
   python scripts/readiness_check.py --base-url http://127.0.0.1:8008 --namespace demo

3. 导入样例知识
   python scripts/ingest_demo.py --base-url http://127.0.0.1:8008 --namespace demo

4. 完整演示并生成报告
   python scripts/interview_demo.py --base-url http://127.0.0.1:8008 --namespace interview_demo
