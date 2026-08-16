# mini_smolagents

从零实现的轻量级 Agent 框架（对标 smolagents），支持终端与 Web UI 两种交互方式，核心引擎、多 Agent 协作、RAG 记忆系统均为独立实现，无 LangChain 依赖。

## 核心特性

- **ReAct 推理引擎**：Think → Act → Observe 循环，基于 OpenAI SDK 原生函数调用（function calling）
- **CodeAgent 沙箱执行**：受控代码生成与执行，内置白名单（`ALLOWED_BUILTINS` / `ALLOWED_IMPORTS`）
- **多 Agent 协作**：静态/动态子 Agent 委托，父 Agent 负责任务拆解与结果综合，防偷懒守卫（lazy guard）
- **三重停止机制**：max_steps 步数上限 + 上下文截断 + final_answer 显式终止
- **RAG 记忆系统**：ChromaDB 向量检索（跨会话语义召回）+ JSON 检查点（会话精确恢复）
- **Web UI**：React + Vite + TypeScript 前端，SSE 流式渲染每一步推理/工具调用过程
- **Docker Compose 一键部署**：frontend + backend + ChromaDB 三容器

## 架构

```
Docker Compose
├── frontend (React+Vite :3000)   ── nginx 反代 ──> backend:8000
├── backend  (FastAPI :8000)
│   ├── /api/chat/stream    # SSE 流式 Agent 执行
│   ├── /api/agents         # Agent 角色列表
│   ├── /api/memory/search  # RAG 语义检索
│   └── /api/sessions       # 会话管理
└── chromadb (:8001)        # 向量存储
```

## 快速开始

### Docker 部署

```bash
cp .env.example .env   # 填入 OPENAI_API_KEY 等
docker compose up --build
```

访问 http://localhost:3000

### 本地开发

```bash
pip install -e .
python examples/basic.py        # 单 Agent 演示
python examples/code_team.py    # 多 Agent 协作演示
```

## 目录结构

```
mini_smolagents/    # 核心框架库
├── agent.py        # ReAct 循环 / CodeAgent / 委托
├── a2a.py          # Agent-to-Agent 协作
├── tools.py        # 工具注册与解析
├── memory.py       # 记忆存储策略
├── llm.py          # LLM 客户端封装
backend/            # FastAPI 服务
frontend/           # React Web UI
tests/              # 单元测试（agent/memory/a2a/stream）
docs/               # PRD / 迭代记录
```

## 学习参考

- [smolagents](https://github.com/huggingface/smolagents)（架构对标）
- [hello-agents](https://github.com/datawhalechina/hello-agents)（入门教程）
