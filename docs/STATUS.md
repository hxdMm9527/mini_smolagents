# Phase 3 阶段性文档（上下文恢复点）

> 保存时间：2026-08-11。用途：压缩上下文后恢复工作现场。

## 1. 项目状态总览

项目：`mini_smolagents`（`D:\Projects\Discussion_Idea\mini_smolagents`）
GitHub：`https://github.com/hxdMm9527/mini_smolagents`。已配置 **SSH over 443**（`~/.ssh/config`：`HostName ssh.github.com` + `Port 443`），push 正常。

**Phase 3 目标**：Web UI + RAG 记忆 + A2A 协议 + Docker 部署。

| 步骤 | 内容 | 状态 |
|---|---|---|
| 1 | `memory.py`（EpisodicMemory + Checkpoint + should_store） | ✅ 完成 |
| 2 | `agent.py`（run_stream + memory 集成） | ✅ 完成 |
| 3 | `a2a.py`（AgentCard/Task/Artifact/AgentRegistry） | ✅ 完成 |
| 4 | `agent.py` 路由改走 AgentRegistry + 动态发现 | ✅ 完成 |
| 5 | `backend/`（FastAPI + SSE + agents_config） | ✅ 完成（commit `7e1d119`） |
| 6 | `frontend/`（React + Vite） | ✅ 完成（commit `6836c36`） |
| 7 | Docker 编排 | ✅ 完成（backend 单容器；frontend 尚未容器化） |

最近 commit（均已 push 到 GitHub）：
- `2e30ec1` Phase 3 steps 1-4 — memory / run_stream / A2A / 动态发现
- `7e1d119` Phase 3 step 5 — FastAPI backend + Docker 部署
- `6836c36` Phase 3 step 6 — React frontend + 助手主 Agent + token 流式

## 2.5 接口设计定稿（Step 5 讨论结果）

**SSE 事件协议**（POST `/api/chat/stream`，`{agent_id, message, session_id?}`）：

```
data: {"type":"session","session_id":"..."}             # 后端生成 session_id
data: {"type":"memory","hits":[...]}                    # 检索到的历史记忆（可选）
data: {"type":"step","agent":"PM","step":1,"max_steps":10}
data: {"type":"thought","agent":"PM","content":"..."}
data: {"type":"action","agent":"PM","tool":"developer","args":{...},"delegation_id":"uuid"}  # ← 子 Agent 调用
data: {"type":"step","agent":"developer","delegation_id":"uuid","step":1,...}   # ← 子 Agent 事件透传
data: {"type":"done","agent":"developer","delegation_id":"uuid","content":"..."} # 子 Agent 结束
data: {"type":"result","agent":"PM","content":"...","delegation_id":"uuid"}     # 父窗口显示子结果
data: {"type":"done","agent":"PM","content":"...","stored":true,"session_id":"..."}
data: {"type":"end"}
```

- 子 Agent 调用**嵌套透传**：父 run_stream 里 `_is_agent_call` 分支，`yield from _stream_delegate(...)`，子 Agent 的每个事件带 `delegation_id` + `agent` 字段透传
- 前端规则：有 `delegation_id` 的事件进子 Agent 小窗口（锚定触发它的 action 消息），无则进主窗口
- `create_sub_agent` 也走嵌套透传（`_prepare_sub_agent` 守卫 + `_finalize_sub_result`）
- SSE 用徒手 `StreamingResponse`（fastapi 自带，无 sse-starlette）
- 端点：`/api/chat/stream`、`/api/agents`、`/api/memory/search`、`/api/health`

**后端文件**：
```
backend/
├── main.py            # FastAPI + CORS 全开 + 路由挂载
├── agents_config.py   # 共享 REGISTRY + MEMORY + PM/developer/reviewer（从 code_team.py 提取）
├── api/{chat,agents,memory}.py
├── requirements.txt   # openai/ddgs/dotenv/rich/chromadb/fastapi/uvicorn
└── Dockerfile         # python:3.12-slim
docker-compose.yml     # 单服务 backend，env_file=.env + chroma_db 卷 + 模型缓存卷
```

**Docker 部署**：`docker compose up -d` → http://127.0.0.1:8000
- 验证：health ok、4 角色（助手/PM/developer/reviewer）、SSE 中文正常、memory 事件命中历史（chroma_db 卷持久化生效）
- 模型缓存卷：宿主 `~/.cache/chroma` → 容器 `/root/.cache/chroma`，避免容器内重下 79MB
- 源码卷挂载 + `uvicorn --reload`：改 Python 代码自动重载，无需 `--build`

**Step 6 补充实现**（相对本协议的新增）：
- **token 级打字机**：thought 内容经 `_generate_stream` 逐 token 下发 `{"type":"token","content":...}`，前端按 token 流式渲染
- **空转兜底**：连续 2 步无 tool_calls 直接结束（idle guard），避免模型自说自话空转
- **中文工具名过滤**：`_build_tools_schema` 用 `re.fullmatch(r"[a-zA-Z0-9_-]+")` 过滤，修复 DeepSeek 对中文工具名报 400 的 bug
- **主 Agent「助手」**：中文名，通用型 MAIN_PROMPT，自己决定如何用 registry 里的子 Agent 完成任务；`/api/agents` 默认选中它
- 4 角色：**助手**（主）+ PM + developer + reviewer（agents_config.py 共享 REGISTRY + MEMORY）

**遗留小问题**：PM 初始无 description（已修，加在 agents_config.py 的 Agent 构造参数里）

## 2. 环境

- Windows，Python 3.12.7，Node v24.15.0，npm 11.12.1
- Docker Desktop 29.4.3 + WSL2 Ubuntu-24.04（可用，Step 7 已验证）
- LLM：DeepSeek，`OPENAI_BASE_URL=https://api.deepseek.com/v1`，key 在 `.env`（`DEEPSEEK_API_KEY`）
- 依赖已装：openai, ddgs, python-dotenv, rich, chromadb（1.5.9）, fastapi（0.141.1）, uvicorn（0.48.0）
- **注意**：SSE 用徒手 `StreamingResponse`，**不装** sse-starlette
- **注意**：ChromaDB 首次用会下载 79MB 的 all-MiniLM-L6-v2 模型（已下载缓存，不慢）
- **注意**：PowerShell 写文件会破坏 UTF-8 中文，用 write/edit 工具而非 PowerShell 重写
- **注意**：Windows 上 ChromaDB 锁文件，临时目录清理需 `ignore_cleanup_errors=True`
- **前端**：Vite + React 19 + TS；`vite.config.ts` 设 `host: '127.0.0.1'`（修 IPv6 ::1 歧义）+ proxy `/api`→8000；`dev.ps1` 一键启动（uvicorn + vite）
- **热重载**：backend 走 `uvicorn --reload`，frontend 走 vite HMR，均自动生效

## 3. 已实现的核心接口（代码层）

### 3.1 `agent.py` — run_stream 事件协议

`Agent.run_stream(task)` 是 generator，yield 以下事件（供 Web SSE 直接用）：

```json
{"type":"step","step":1,"max_steps":10}
{"type":"thought","content":"思考内容"}
{"type":"action","tool":"web_search","args":{"query":"..."}}
{"type":"result","content":"工具返回"}
{"type":"note","content":"达到最大步数，正在总结..."}
{"type":"done","content":"最终答案","stored":true}
```

- `done.stored` = 是否存入长期记忆（方案 B 判断）
- `run()` 改为消费 run_stream，负责 Rich 打印 + 返回最终结果
- 超时/失败路径：yield note + done（stored=false），不存记忆
- memory 注入：run 开头 `_inject_memory(task)` 检索 top_k=3 拼进 system prompt

### 3.2 `agent.py` — 工具调用路由

- `_build_tools_schema()`：静态工具 + registry 中未重复的 Agent 卡片（动态发现）
- `_execute_tool(name, args)`：静态工具 → `registry.find(name)` delegate → 否则 `"Error: unknown tool"`

### 3.3 `memory.py` — 记忆层

- `EpisodicMemory(collection_name, persist_dir)`：`add(task,result)` / `search(query,top_k)` / `clear()` / `count()`
- `Checkpoint(base_dir)`：`save/load/list_sessions/delete(session_id)`，文件在 `.memory/{session_id}.json`
- `should_store(task, result, min_length=10)`：过短 / 含失败标记（error/timed out/traceback/失败/异常）→ False
- 方案 B：只存 final_answer 正常路径的结果

### 3.4 `a2a.py` — A2A 协议

- `AgentCard(name, description, capabilities[], tools[])`
- `Task(description, target_agent, task_id, context, parent_agent)`
- `Artifact(task_id, status, content, error)`，status ∈ success/fail
- `AgentRegistry`：`register(agent, capabilities)` / `find(name)` / `get_agent` / `list_capabilities` / `list_cards` / `delegate(task)`
- `delegate` 不抛异常：Agent 不存在或执行失败 → 返回 fail Artifact

## 4. Step 5 backend 待设计接口（用户正在讨论的焦点）

初步方案（PRD §4.3）：

| 端点 | 方法 | 入参 | 返回 |
|---|---|---|---|
| `/api/chat/stream` | POST | `{agent_id, message, session_id?}` | SSE 流（step→thought→action→result→done） |
| `/api/agents` | GET | — | 可用 Agent 角色列表（来自 AgentRegistry） |
| `/api/memory/search` | POST | `{query}` | 语义检索结果 |

**待讨论点**：
- SSE 数据格式是否直接复用 run_stream 事件 dict（我建议是，JSON 逐行）
- `agent_id` 如何与 registry 中的 Agent 对应（registry 全局共享一个实例？）
- 是否需要支持 `create_sub_agent` 动态 Agent 的展示（A2A 调用链）
- 会话管理：session_id 由前端生成还是后端生成
- 是否把 memory 检索结果一并返回给前端（还是只注入 system prompt）

## 5. 文件结构现状

```
mini_smolagents/
├── mini_smolagents/
│   ├── __init__.py      # 导出全部公共类
│   ├── agent.py         # Agent/CodeAgent + run_stream + token 流 + registry 路由
│   ├── a2a.py           # AgentCard/Task/Artifact/AgentRegistry
│   ├── memory.py        # EpisodicMemory/Checkpoint/should_store
│   ├── llm.py           # OpenAIModel
│   ├── default_tools.py # web_search/python_interpreter/final_answer
│   ├── tools.py         # @tool 装饰器
│   └── types.py         # Tool dataclass
├── backend/             # Step 5 FastAPI
│   ├── main.py          # FastAPI + CORS 全开 + 路由挂载
│   ├── agents_config.py # 共享 REGISTRY + MEMORY + 4 角色（助手/PM/developer/reviewer）
│   ├── api/{chat,agents,memory}.py   # chat.py 为同步 generator
│   ├── requirements.txt # openai/ddgs/dotenv/rich/chromadb/fastapi/uvicorn
│   └── Dockerfile       # python:3.12-slim
├── frontend/            # Step 6 React + Vite
│   ├── src/App.tsx + main.tsx + useStreamChat.ts + groupEvents.ts + types.ts
│   ├── src/components/{Header,ChatPanel,StreamBlock,SubAgentWindow,MemoryPanel}.tsx
│   └── vite.config.ts   # host 127.0.0.1 + proxy /api→8000
├── docker-compose.yml   # 单服务 backend，env_file=.env + chroma_db 卷 + 模型缓存卷 + --reload
├── dev.ps1              # 一键启动 backend + frontend
├── tests/
│   ├── test_memory.py   # 6+1 用例
│   ├── test_stream.py   # 5 用例（含 idle guard）
│   └── test_a2a.py      # 12 用例
├── examples/
│   ├── code_team.py     # PM+dev+reviewer demo（走新 registry 路由）
│   ├── basic.py / dome.py / delegate_test.py
├── docs/PRD.md          # 完整 Phase 3 设计
└── pyproject.toml       # 依赖含 chromadb
```

## 6. 当前状态与遗留项

Phase 3 全部 7 步已完成并 push（SSH over 443，远程 = 本地 `6836c36`，working tree clean）。

**遗留项**（非阻塞）：
1. `tests/__pycache__/test_agent.cpython-312-pytest-9.1.1.pyc` 被误跟踪在 git 索引，应从索引移除并加进 `.gitignore`
2. frontend 尚未进 docker-compose（Step 7 只容器化 backend）；如要容器化需 nginx 反代 backend
3. 后续：Phase 4 开发前可更新本文件
