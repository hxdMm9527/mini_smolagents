import io
import json
import re
import sys
import threading
import time
import uuid
from types import SimpleNamespace
from contextlib import redirect_stderr, redirect_stdout

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .default_tools import ALLOWED_BUILTINS, ALLOWED_IMPORTS, _safe_import, final_answer as _FINAL_ANSWER_TOOL
from .memory import should_store
from .types import Tool

SYSTEM_PROMPT = """\
你是一个善于逐步解决问题的助手。你可以使用工具调用来完成任务。
当你有最终答案时，调用 `final_answer` 工具。
不要在相同的参数下重复调用同一个工具。
如果你使用了 `create_sub_agent`，你仍然是最终答案的负责人。
不要把用户任务原封不动转发给子助手——你必须亲自分析、拆解后再委托。
子助手的返回结果不能直接作为 final_answer，你需要亲自综合或验证。\
"""


class Agent:
    def __init__(self, model, tools, max_steps=10, max_messages=30, stream=False, name=None, description=None, managed_agents=None, allow_delegation=True, system_prompt=None, memory=None, checkpoint=None, session_id=None, registry=None):
        self.model = model
        self.max_steps = max_steps
        self.max_messages = max_messages
        self.stream = stream
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.console = Console(force_terminal=True)
        self.name = name
        self.description = description
        self.memory = memory
        self.checkpoint = checkpoint
        self.session_id = session_id
        self.registry = registry
        self.tools = {}
        self._sub_results: dict[str, str] = {}
        self._delegation_count = 0

        for t in tools:
            self.tools[t.name] = t

        if managed_agents:
            for sub in managed_agents:
                sub_name = sub.name
                sub_desc = sub.description
                if not sub_name or not sub_desc:
                    raise ValueError("每个被管理的 Agent 必须有 name 和 description")
                if sub_name in self.tools:
                    raise ValueError(f"被管理的 Agent 名称 '{sub_name}' 与已有工具重名")
                self._ensure_registry().register(sub)
                self.tools[sub_name] = self._build_sub_agent_tool(sub_name, sub_desc)

        if allow_delegation:
            self.tools["create_sub_agent"] = self._build_create_sub_agent_tool()

        if "final_answer" not in self.tools:
            self.tools["final_answer"] = _FINAL_ANSWER_TOOL

    def _ensure_registry(self):
        if self.registry is None:
            from .a2a import AgentRegistry
            self.registry = AgentRegistry()
        return self.registry

    def _delegate(self, target: str, task: str) -> str:
        from .a2a import Task
        artifact = self._ensure_registry().delegate(Task(description=task, target_agent=target))
        if artifact.status == "success":
            return artifact.content
        return f"子助手执行失败: {artifact.error}"

    def _build_sub_agent_tool(self, sub_name: str, sub_desc: str) -> Tool:
        agent_self = self

        def _call(task: str = "") -> str:
            return agent_self._delegate(sub_name, task)

        return Tool(
            name=sub_name,
            description=sub_desc,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": f"给 {sub_name} 的详细任务描述",
                    }
                },
                "required": ["task"],
            },
            func=_call,
        )

    def _build_create_sub_agent_tool(self):
        agent_self = self

        def create_sub_agent(name: str, task: str, tools: str = "") -> str:
            agent, final_task, err = agent_self._prepare_sub_agent(name, task, tools)
            if err:
                return err
            result = agent_self._delegate(name, final_task)
            agent_self._sub_results[name] = result
            return agent_self._finalize_sub_result(result)

        return Tool(
            name="create_sub_agent",
            description=(
                "创建一个临时助手来执行需要独立研究的复杂子任务。"
                "重要：你必须亲自拆解任务给子助手，不能把用户任务原封不动转发。"
                "子助手返回结果后，你必须亲自综合或验证，不能直接作为最终答案。"
                "适合将复杂任务拆分给专门的助手处理。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "临时助手的名称"},
                    "task": {"type": "string", "description": "给临时助手的详细任务描述"},
                    "tools": {
                        "type": "string",
                        "description": "助手可用的工具名称，逗号分隔。留空则继承主 Agent 的所有工具。",
                    },
                },
                "required": ["name", "task"],
            },
            func=create_sub_agent,
        )

    def _build_tools_schema(self):
        schema = []
        for t in self.tools.values():
            schema.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        if self.registry:
            for card in self.registry.list_cards():
                if card.name in self.tools:
                    continue
                if not re.fullmatch(r"[a-zA-Z0-9_-]+", card.name):
                    # 中文名 Agent 不作为可调用工具暴露（DeepSeek 要求工具名 ^[a-zA-Z0-9_-]+$）
                    continue
                schema.append({
                    "type": "function",
                    "function": {
                        "name": card.name,
                        "description": (
                            f"{card.description}"
                            + (f"。能力：{', '.join(card.capabilities)}" if card.capabilities else "")
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": f"给 {card.name} 的详细任务描述",
                                }
                            },
                            "required": ["task"],
                        },
                    },
                })
        return schema

    def _execute_tool(self, name: str, args: dict) -> str:
        if name in self.tools:
            return self.tools[name].func(**args)
        if self.registry and self.registry.find(name):
            return self._delegate(name, args["task"])
        return f"Error: unknown tool '{name}'"

    def _get_trimmed_messages(self, messages):
        if len(messages) <= self.max_messages:
            return messages
        return [messages[0], messages[1]] + messages[-(self.max_messages - 2):]

    def _summarize_messages(self, messages):
        parts = []
        for msg in messages:
            if msg["role"] == "tool" and msg.get("content"):
                parts.append(f"工具返回: {msg['content'][:800]}")
            elif msg["role"] == "assistant" and msg.get("content"):
                parts.append(f"分析: {msg['content'][:500]}")

        if not parts:
            return "(任务未完成，无有效信息)"

        summary_prompt = (
            "请基于以下步骤记录，总结已完成的工作和获得的关键信息。只提取有价值的事实数据，忽略错误和空结果。用中文简要回答。\n\n"
            + "\n---\n".join(parts)
        )
        summary_msgs = [
            {"role": "system", "content": "你是一个善于总结信息的助手。"},
            {"role": "user", "content": summary_prompt},
        ]
        try:
            resp = self.model.generate(summary_msgs)
            return resp.choices[0].message.content
        except Exception:
            return "\n".join(parts[:3]) + "\n\n(超时，以上为部分结果)"

    def _retrieve_memory(self, task: str) -> list[dict]:
        if not self.memory:
            return []
        try:
            return self.memory.search(task, top_k=3)
        except Exception:
            return []

    def _inject_memory(self, task: str) -> str:
        history = self._retrieve_memory(task)
        if not history:
            return self.system_prompt
        mem_lines = "\n".join(f"- {h['task']} → {h['document'][:200]}" for h in history)
        return f"{self.system_prompt}\n\n[相关历史记忆，供参考：]\n{mem_lines}"

    def _event(self, type_: str, **kw) -> dict:
        ev = {"type": type_, "agent": self.name}
        ev.update(kw)
        return ev

    def _is_agent_call(self, tool_name: str) -> bool:
        return tool_name == "create_sub_agent" or bool(self.registry and self.registry.find(tool_name))

    def _stream_delegate(self, tool_name: str, args: dict, delegation_id: str):
        """嵌套透传子 Agent 的 run_stream，返回其最终结果。"""
        if tool_name == "create_sub_agent":
            agent, final_task, err = self._prepare_sub_agent(
                args.get("name", ""), args.get("task", ""), args.get("tools", "")
            )
            if err:
                return err
            result = ""
            for ev in agent.run_stream(final_task, delegation_id=delegation_id):
                yield ev
                if ev["type"] == "done":
                    result = ev["content"]
            result = self._finalize_sub_result(result)
            self._sub_results[agent.name] = result
            return result

        agent = self.registry.get_agent(tool_name) if self.registry else None
        if agent is None:
            return f"子助手执行失败: Agent '{tool_name}' not registered"
        result = ""
        for ev in agent.run_stream(args.get("task", ""), delegation_id=delegation_id):
            yield ev
            if ev["type"] == "done":
                result = ev["content"]
        return result

    def _finalize_sub_result(self, result: str) -> str:
        result += "\n\n[系统提示：请验证以上子助手的结果，给出你自己的综合分析后再调用 final_answer。]"
        if "timed out" in result.lower() or "error" in result.lower():
            result += "\n\n[警告：子助手执行遇到问题。请不要再次创建同名子助手重试——请换用其他方式自己完成任务。]"
        return result

    def _prepare_sub_agent(self, name: str, task: str, tools_str: str = ""):
        """守卫检查 + 创建/复用子 Agent。返回 (agent, final_task, error)。"""
        self._delegation_count += 1
        if self._delegation_count > 3:
            return None, None, "已达到最大委托次数（3次）。请你自己直接完成任务，不要再次创建子助手。"

        if hasattr(self, "_original_task") and task.strip() == self._original_task.strip():
            return None, None, "错误：不能把用户任务原封不动转发给子助手。请先自己分析、拆解后再委托。"

        tool_names = [t.strip() for t in tools_str.split(",") if t.strip()]
        sub_tools = []
        for tn in tool_names:
            if tn in self.tools and tn not in ("create_sub_agent", "final_answer"):
                sub_tools.append(self.tools[tn])
        if not sub_tools:
            sub_tools = [
                t for t in self.tools.values()
                if t.name not in ("create_sub_agent", "final_answer")
            ]

        if name in self._sub_results and self._sub_results[name]:
            task = (
                f"{task}\n\n"
                f"[上次查找结果供参考，请在此基础上继续，不要重复搜索已有信息：]\n"
                f"{self._sub_results[name]}"
            )

        registry = self._ensure_registry()
        agent = registry.get_agent(name)
        if agent is None:
            agent = Agent(
                model=self.model,
                tools=sub_tools,
                name=name,
                max_steps=min(5, self.max_steps),
                max_messages=self.max_messages,
                allow_delegation=False,
                registry=registry,
            )
            registry.register(agent)
        return agent, task, None

    def _save_checkpoint(self):
        if self.checkpoint and self.session_id:
            try:
                self.checkpoint.save(self.session_id, self._last_messages)
            except Exception:
                pass

    def _generate_stream(self, messages, tools_schema, delegation_id=None):
        """流式调用 LLM。yield token 事件，最终返回组装好的 message。

        - 有 generate_stream 时：逐 token yield {"type":"token","content":...}
        - 无流式接口（测试 FakeModel）时：一次性 yield 完整 thought
        """
        if not hasattr(self.model, "generate_stream"):
            response = self.model.generate(messages, tools_schema)
            msg = response.choices[0].message
            text = msg.content or ""
            if text:
                ev = self._event("thought", content=text)
                if delegation_id:
                    ev["delegation_id"] = delegation_id
                yield ev
            return msg

        stream = self.model.generate_stream(messages, tools_schema)
        text = ""
        tool_calls = {}
        order = []

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                text += delta.content
                ev = self._event("token", content=delta.content)
                if delegation_id:
                    ev["delegation_id"] = delegation_id
                yield ev
            for tc in delta.tool_calls or []:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    order.append(idx)
                if tc.id:
                    tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc.function.arguments

        tcs = None
        if order:
            tcs = [
                SimpleNamespace(
                    id=tool_calls[i]["id"],
                    type="function",
                    function=SimpleNamespace(
                        name=tool_calls[i]["function"]["name"],
                        arguments=tool_calls[i]["function"]["arguments"],
                    ),
                )
                for i in order
            ]
        return SimpleNamespace(content=text or None, tool_calls=tcs)

    def run_stream(self, task: str, delegation_id: str | None = None):
        """核心执行逻辑的 generator。只 yield 事件，不打印。

        事件类型：
        {"type":"step","step":n,"max_steps":m}
        {"type":"memory","hits":[...]}
        {"type":"thought","content":...}
        {"type":"action","tool":...,"args":{...}}
        {"type":"result","content":...}
        {"type":"note","content":...}
        {"type":"done","content":...,"stored":bool}

        子 Agent 事件透传：事件带 "agent" 和 "delegation_id" 字段。
        """
        # ponytail: 配合 create_sub_agent 守卫 1 使用。取消注释下方守卫后启用此行
        self._original_task = task
        history = self._retrieve_memory(task)
        if history:
            yield self._event("memory", hits=history)
        messages = [
            {"role": "system", "content": self._inject_memory(task)},
            {"role": "user", "content": task},
        ]
        self._last_messages = messages

        idle_steps = 0
        for step in range(1, self.max_steps + 1):
            yield self._event("step", step=step, max_steps=self.max_steps)

            trimmed = self._get_trimmed_messages(messages)
            tools_schema = self._build_tools_schema()
            msg = yield from self._generate_stream(trimmed, tools_schema, delegation_id=delegation_id)
            text = msg.content or ""

            if not msg.tool_calls:
                idle_steps += 1
                messages.append({"role": "assistant", "content": text})
                # ponytail: 连续空转兜底，避免问候语/无任务时空转到 max_steps
                if idle_steps >= 2:
                    note = f"[{self.name}] 连续多步未调用工具，直接结束。"
                    final = text or self._summarize_messages(messages)
                    yield self._event("note", content=note)
                    yield self._event("done", content=final, stored=False, session_id=self.session_id)
                    self._save_checkpoint()
                    return
                continue
            idle_steps = 0

            tool_calls_dict = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": tool_calls_dict,
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                raw_args = tc.function.arguments
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

                if self._is_agent_call(tool_name):
                    did = str(uuid.uuid4())
                    yield self._event("action", tool=tool_name, args=args, delegation_id=did)
                    result = ""
                    try:
                        result = yield from self._stream_delegate(tool_name, args, did)
                    except Exception as e:
                        result = f"Error: {type(e).__name__}: {e}"
                    yield self._event("result", content=str(result), delegation_id=did)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
                    continue

                yield self._event("action", tool=tool_name, args=args)

                result = None
                for attempt in range(1, 4):
                    try:
                        result = self._execute_tool(tool_name, args)
                        break
                    except Exception as e:
                        if attempt == 3:
                            result = f"Error after 3 retries: {type(e).__name__}: {e}"

                yield self._event("result", content=str(result))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

                if tool_name == "final_answer":
                    final = str(result)
                    stored = False
                    if self.memory and should_store(task, final):
                        self.memory.add(task, final)
                        stored = True
                    yield self._event("done", content=final, stored=stored, session_id=self.session_id)
                    self._save_checkpoint()
                    return

        summary = self._summarize_messages(self._last_messages)
        yield self._event("note", content=f"[{self.name}] 达到最大步数，正在总结已有结果...")
        yield self._event("done", content=summary, stored=False, session_id=self.session_id)
        self._save_checkpoint()

    def run(self, task: str) -> str:
        result = ""
        for event in self.run_stream(task):
            self._print_event(event)
            if event["type"] == "done" and "delegation_id" not in event:
                result = event["content"]
        return result

    def _print_event(self, event: dict):
        etype = event["type"]
        who = event.get("agent", self.name)
        did = f" (sub:{event['delegation_id'][:8]})" if event.get("delegation_id") else ""
        if etype == "step":
            self.console.print(Rule(f"[{who}] Step {event['step']}/{event['max_steps']}{did}", style="bold blue"))
        elif etype == "thought":
            text = event["content"]
            if self.stream:
                with Live("", console=self.console, refresh_per_second=60) as live:
                    for i in range(1, len(text) + 1, 2):
                        live.update(Markdown(text[:i]))
                        time.sleep(0.01)
                    live.update(Markdown(text))
            else:
                self.console.print(Markdown(text))
        elif etype == "token":
            # ponytail: 终端不做逐 token 打字机，直接忽略增量（run() 靠 done 取结果）
            pass
        elif etype == "action":
            action_text = Text(f"[{who}] Action: {event['tool']}", style="bold yellow")
            action_text.append(f"\nArgs: {_trunc(str(event['args']), 200)}", style="dim")
            self.console.print(Panel(action_text, border_style="yellow"))
        elif etype == "result":
            self.console.print(Panel(Text(str(event["content"])[:500], style="green"), border_style="green", title=f"[{who}] Result"))
        elif etype == "note":
            self.console.print(Panel(Text(event["content"], style="orange3"), border_style="orange3"))
        elif etype == "done":
            self.console.print(Panel(Text(str(event["content"]), style="bold gold1"), border_style="gold1", title=f"[{who}] Done"))


class _FinalAnswer(Exception):
    def __init__(self, result):
        self.result = result


class _StopExec(BaseException):
    pass


CODE_SYSTEM_PROMPT = """\
你是一个善于编写 Python 代码来解决问题的助手。
每步写一段 Python 代码，用 <code> 和 </code> 包裹。

你可以直接调用以下 Python 函数：
{tools_description}

允许导入的库：{imports}

得到最终答案时，调用 final_answer(答案)。
"""


def _extract_code(text: str) -> str:
    match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _trunc(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class CodeAgent(Agent):
    def __init__(self, model, tools, max_steps=8, max_messages=30, additional_imports=None, name=None, description=None, managed_agents=None):
        super().__init__(model, tools, max_steps, max_messages, name=name, description=description, managed_agents=managed_agents)
        self.authorized_imports = list(set(ALLOWED_IMPORTS) | set(additional_imports or []))

    def _build_sandbox(self):
        fa = {"value": None}

        def _fa(value):
            fa["value"] = str(value)
            raise _StopExec()

        g = {
            "__builtins__": {**ALLOWED_BUILTINS, "__import__": _safe_import},
        }
        for t in self.tools.values():
            g[t.name] = t.func
        g["final_answer"] = _fa
        return g, fa

    def _run_code(self, code: str, sandbox: dict, fa: dict) -> tuple[str, str]:
        result: dict = {"output": "", "error": "", "timed_out": False}

        def _run():
            f = io.StringIO()
            try:
                with redirect_stdout(f), redirect_stderr(f):
                    exec(code, sandbox)
                result["output"] = f.getvalue().strip() or "(no output)"
            except _StopExec:
                result["output"] = f.getvalue().strip() or "(no output)"
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=30)
        if t.is_alive():
            result["timed_out"] = True

        if fa["value"] is not None:
            return ("final_answer", fa["value"])

        if result["timed_out"]:
            return ("error", "Error: code execution timed out (30-second limit).")
        if result["error"]:
            return ("error", f"Error: {result['error']}")
        return ("output", result["output"])

    def run(self, task: str) -> str:
        tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        imports = ", ".join(self.authorized_imports)
        sys_prompt = CODE_SYSTEM_PROMPT.format(tools_description=tool_descs, imports=imports)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task},
        ]
        sandbox, fa = self._build_sandbox()

        for step in range(1, self.max_steps + 1):
            self.console.print(Rule(f"[{self.name}] Step {step}/{self.max_steps}", style="bold blue"))

            if self.stream:
                response = self.model.generate(self._get_trimmed_messages(messages))
                msg = response.choices[0].message
                full_text = msg.content or ""
                if full_text:
                    with Live("", console=self.console, refresh_per_second=60) as live:
                        for i in range(1, len(full_text) + 1, 2):
                            live.update(Text(full_text[:i]))
                            time.sleep(0.01)
                        live.update(Text(full_text))
            else:
                response = self.model.generate(self._get_trimmed_messages(messages))
                msg = response.choices[0].message
                full_text = msg.content or ""
                if full_text:
                    self.console.print(Text(full_text[:500]))

            code = _extract_code(full_text)
            messages.append({"role": "assistant", "content": full_text})

            status, value = self._run_code(code, sandbox, fa)
            value_short = _trunc(value, 500)
            if status == "error":
                self.console.print(Panel(Text(value_short, style="red"), border_style="red", title=f"[{self.name}] Error"))
            elif status == "final_answer":
                self.console.print(Panel(Text(value_short, style="bold gold1"), border_style="gold1", title=f"[{self.name}] Code output"))
            else:
                self.console.print(Panel(Text(value_short, style="green"), border_style="green", title=f"[{self.name}] Code output"))
            messages.append({"role": "user", "content": value})

            if status == "final_answer":
                self.console.print(Panel(Text(value, style="bold gold1"), border_style="gold1", title=f"[{self.name}] Done"))
                return value

        self.console.print(Panel(Text(f"[{self.name}] 达到最大步数，正在总结已有结果...", style="orange3"), border_style="orange3"))
        return self._summarize_messages(self._last_messages)
