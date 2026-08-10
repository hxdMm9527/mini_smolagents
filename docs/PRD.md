# PRD: mini_smolagents Phase 3 — Web UI + RAG 记忆系统

## 1. 背景

Phase 1：ReAct Agent + 工具系统 + LLM 集成。

Phase 2：CodeAgent + 上下文截断 + Rich 日志 + 静态/动态多 Agent 委托 + 防偷懒守卫。

**Phase 2 遗留问题**：
- Agent 无记忆：每次 `run()` 全新对话，关闭程序全部丢失
- 只能在终端运行：需要命令行交互，无法通过浏览器使用
- 流式展示未解决：终端依赖 Rich `Live`，Windows 不支持原地刷新
- 无 RAG：无法跨会话检索历史知识

## 2. 目标

| 目标 | 说明 |
|---|---|
| **Web 聊天界面** | React 页面，选择 Agent 角色、流式查看输出、输入对话 |
| **终端流式 → 页面流式** | 页面通过 SSE 实时展示 Agent 每步输出，解决终端渲染限制 |
| **RAG 记忆系统** | ChromaDB 向量检索 + JSON 检查点，跨会话持久化 |
| **Docker 一键部署** | `docker-compose up` 启动前端 + 后端 + ChromaDB |

## 3. 架构

```
Docker Compose
├── frontend (React+Vite :3000)
│   └── nginx 反代 → backend:8000
├── backend (FastAPI :8000)
│   ├── /api/chat/stream    ← SSE 流式 Agent 执行
│   ├── /api/agents          ← 可用 Agent 角色列表
│   └── /api/memory/search   ← RAG 语义检索
└── chromadb (:8001)
    └── 向量存储
```

## 4. 功能模块

### 4.1 RAG 记忆系统

| 类 | 存储 | 检索方式 | 生命周期 |
|---|---|---|---|
| `EpisodicMemory` | ChromaDB 向量库 | `search(query, top_k)` 语义检索 | 永久 |
| `Checkpoint` | JSON 文件 `.memory/{session_id}.json` | `load(session_id)` 精确恢复 | 手动删除前永久 |
| `self._last_messages`（已有） | 内存 | 全量传给 LLM | 当前会话 |

**Agent 自动集成**：
- `run()` 开头：自动检索相关历史 → 注入 system prompt
- `run()` 结尾：自动存入 ChromaDB + 更新检查点

### 4.2 流式输出

Agent 新增 `run_stream()` generator 方法：

```python
def run_stream(self, task):
    yield {"type": "step", "step": 1, "max_steps": 10}
    yield {"type": "thought", "content": "我需要..."}
    yield {"type": "action", "tool": "web_search", "args": {...}}
    yield {"type": "result", "content": "..."}
    yield {"type": "done", "content": "最终答案"}
```

`run()` 改为消费 `run_stream()`——逻辑只有一份。不改已有 `run()` 行为。

### 4.3 后端 API

| 端点 | 方法 | 入参 | 返回 |
|---|---|---|---|
| `/api/chat/stream` | POST | `{agent_id, message, session_id?}` | SSE 流（step→thought→action→result→done） |
| `/api/agents` | GET | — | Agent 角色列表 |
| `/api/memory/search` | POST | `{query}` | 语义检索结果 |

SSE 数据格式：

```
data: {"type":"step","step":1,"max_steps":10}
data: {"type":"thought","content":"我需要搜索..."}
data: {"type":"action","tool":"web_search","args":{"query":"北京天气"}}
data: {"type":"result","content":"1. 北京今天15°C..."}
data: {"type":"done","content":"北京今天15°C"}
```

### 4.4 A2A 协议（Agent-to-Agent）

标准化多 Agent 通信，取代现有 `create_sub_agent` 的松散约定。

**数据模型**：

| 结构 | 字段 | 说明 |
|---|---|---|
| `AgentCard` | `name, description, capabilities[], tools[], input_schema` | Agent 的"名片"，注册时提供 |
| `Task` | `task_id, description, context, parent_agent` | 委托任务，标准化入参 |
| `Artifact` | `task_id, status(success/fail/partial), content, error` | 委托结果，标准化出参 |

**AgentRegistry**：
- `register(card)` → 注册 Agent 到目录
- `find(name)` → 按名查找
- `list_capabilities()` → 列出所有能力
- `delegate(task)` → 查找对应 Agent → 执行 → 返回 Artifact

**Web UI 展示**：
- 多 Agent 调用链可视化（树形结构）：PM → [Developer, Reviewer] → PM
- 每个子 Agent 的执行步数、耗时、结果折叠在父 Agent 的步骤中

### 4.5 前端页面

```
┌──────────────────────────────────────────────────┐
│  mini_smolagents              [Agent: PM  ▼]     │
├────────────────────────────┬─────────────────────┤
│                            │  🔍 记忆搜索         │
│  💬 用户: 写邮箱验证        │  搜索框 + 历史列表   │
│                            │                     │
│  🤖 PM: Step 1/10          │  相关记忆:          │
│     📝 思考中...            │  · 邮箱bug讨论      │
│     🔧 调用 developer       │  · regex用法        │
│     ✅ 返回代码...           │  · 测试用例         │
│                            │                     │
│  🤖 PM: Step 2/10          │                     │
│     🔧 调用 reviewer        │                     │
│     ✅ 审核通过              │                     │
│                            │                     │
│  ✨ 最终交付：完整代码       │                     │
│                            │                     │
│  ┌────────────────────────┐│                     │
│  │ 输入消息...        发送 ││                     │
│  └────────────────────────┘│                     │
└────────────────────────────┴─────────────────────┘
```

**组件树**：

```
App
├── Header（标题 + AgentSelector）
├── ChatPanel
│   ├── ChatBubble（用户消息）
│   ├── StreamBlock（流式事件动态增长）
│   │   ├── StepBadge
│   │   ├── ThoughtCard
│   │   ├── ActionCard
│   │   └── ResultCard
│   └── ChatInput
└── MemoryPanel（RAG 搜索 + 历史列表）
```

**流式消费**（不用 EventSource，它不支持 POST）：

```typescript
const response = await fetch("/api/chat/stream", {
  method: "POST",
  body: JSON.stringify({ agent_id, message }),
});
const reader = response.body!.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  // 解析 SSE，逐条追加到 React state → 界面动态渲染
}
```

## 5. 实施步骤

```
1. memory.py     → EpisodicMemory + Checkpoint（独立，先做）
2. agent.py      → 新增 run_stream() + 集成 memory（依赖 1）
3. a2a.py        → AgentCard/Task/Artifact + AgentRegistry（依赖 2）
4. agent.py(补)  → create_sub_agent 改为走 AgentRegistry（依赖 3）
5. backend/      → FastAPI + SSE + agents_config（依赖 4）
6. frontend/     → React + Vite + 组件（依赖 5）
7. Docker        → docker-compose 四服务编排
```

**验证路径（每步独立，不跨步）**：

| 步骤 | 验证 |
|---|---|
| 1 | `python test_memory.py`：写入 → 语义搜索 |
| 2 | `python test_stream.py`：迭代 generator → 逐条 print |
| 3 | `python test_a2a.py`：注册 → 委托 → 收到 Artifact |
| 4 | `python code_team.py`：走新协议，行为与旧版一致 |
| 5 | `curl -N POST /api/chat/stream`：逐行 SSE |
| 6 | 浏览器 → 输入 → 流式渲染 + RAG 侧边栏 |
| 7 | `docker-compose up` → 浏览器全栈 |

## 6. 文件结构

```
mini_smolagents/
├── mini_smolagents/
│   ├── agent.py          # +run_stream() + memory 集成
│   ├── memory.py         # EpisodicMemory + Checkpoint（新）
│   ├── a2a.py            # AgentCard/Task/Artifact + AgentRegistry（新）
├── backend/              # 新目录
│   ├── main.py
│   ├── agents_config.py
│   ├── api/
│   │   ├── chat.py
│   │   ├── agents.py
│   │   └── memory.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/             # 新目录
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── AgentSelector.tsx
│   │   │   ├── StreamBlock.tsx
│   │   │   └── MemoryPanel.tsx
│   │   ├── hooks/useStreamChat.ts
│   │   └── types.ts
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml
└── docs/PRD.md
```

## 7. 依赖新增

```
chromadb>=0.4      # RAG 向量记忆
fastapi>=0.100     # 后端 Web 框架
uvicorn>=0.23      # ASGI 服务器
sse-starlette      # SSE 支持
```

## 8. 后续（本次不做）

- 并行委托（多子 Agent 异步并发执行，等待全部结果后综合）
- ProjectBlackboard 共享黑板
- Planning 步骤
- 多 LLM Provider
