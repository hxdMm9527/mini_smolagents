from dataclasses import asdict
import json
import logging
import re
import sys
import uuid

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console

from .a2a import AgentRegistry
from .config import DEFAULT_MAX_STEPS, DEFAULT_WINDOW_SIZE, FACTS_TOP_K, MEMORY_TOP_K, SUB_AGENT_MAX_STEPS, TOOL_RETRY_ATTEMPTS, TRUNC_LONG, TRUNC_MEDIUM, TRUNC_SHORT
from .console import print_event
from .context import ContextComposer, SessionMetadata
from .default_tools import final_answer as _FINAL_ANSWER_TOOL
from .facts import FactsMemory
from .memory import MemoryHit, StorePolicy
from .profile import Profile
from .prompts import SYSTEM_PROMPT
from .types import ModelResponse, Tool

logger = logging.getLogger(__name__)



class Agent:
    def __init__(self, model, tools, max_steps=DEFAULT_MAX_STEPS, window_size=DEFAULT_WINDOW_SIZE, stream=False, name=None, description=None, managed_agents=None, allow_delegation=True, system_prompt=None, memory=None, checkpoint=None, session_id=None, registry=None, store_policy=None, profile_dir=None, facts_memory=None, auto_extract_facts=False):
        self.model = model
        self.max_steps = max_steps
        self.window_size = window_size
        self.composer = ContextComposer()
        self.summary_block = ""
        self._summarized_upto = 0
        self.stream = stream
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.console = Console(force_terminal=True)
        self.name = name
        self.description = description
        self.memory = memory
        self.checkpoint = checkpoint
        self.session_id = session_id
        self.registry = registry
        self.store_policy = store_policy or StorePolicy()
        self.profile_dir = profile_dir
        self._profile = None
        self.facts_memory = facts_memory
        self.auto_extract_facts = auto_extract_facts
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

        if profile_dir:
            self.tools["update_profile"] = self._build_update_profile_tool()

    def _ensure_registry(self):
        if self.registry is None:
            self.registry = AgentRegistry()
        return self.registry

    def _delegate(self, target: str, task: str) -> str:
        artifact = self._ensure_registry().delegate_to(target, task)
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

    def _build_update_profile_tool(self):
        agent_self = self

        def update_profile(name=None, role=None, preferences=None, constraints=None,
                           facts=None, feedback=None, style_prefs=None):
            profile = agent_self._profile
            if profile is None:
                return "档案卡未启用，本次更新被忽略。"
            applied = []
            rejected = []
            if name is not None:
                (applied if profile.set_name(name) else rejected).append("name")
            if role is not None:
                (applied if profile.set_role(role) else rejected).append("role")
            if isinstance(preferences, dict):
                for k, v in preferences.items():
                    (applied if profile.set_preference(k, v) else rejected).append(f"preference[{k}]")
            if isinstance(style_prefs, dict):
                for k, v in style_prefs.items():
                    (applied if profile.set_style_pref(k, v) else rejected).append(f"style_pref[{k}]")
            if constraints:
                (applied if profile.append_constraints(constraints) else rejected).append("constraints")
            if feedback:
                (applied if profile.append_feedback(feedback) else rejected).append("feedback")
            if facts:
                raw = facts if isinstance(facts, list) else [facts]
                items = [str(x).strip() for x in raw if str(x).strip()]
                if agent_self.facts_memory is not None and items:
                    added = sum(1 for f in items if agent_self.facts_memory.add(f))
                    skipped = len(items) - added
                    if added:
                        applied.append(f"facts(写入 {added} 条)")
                    if skipped:
                        rejected.append(f"facts(去重跳过 {skipped} 条)")
                elif items:
                    rejected.append("facts(存储未启用)")
            message = f"档案卡已更新：{', '.join(applied)}" if applied else "档案卡无变更"
            if rejected:
                message += f"；被拒绝：{', '.join(rejected)}"
            return message

        return Tool(
            name="update_profile",
            description=(
                "更新用户档案卡，记住用户的姓名、角色、偏好、约束、事实、反馈和风格偏好。"
                "当你了解到用户的长期信息时调用此工具。只需传入要更新的字段。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "用户姓名（仅首次写入，之后忽略）"},
                    "role": {"type": "string", "description": "用户角色（仅首次写入，之后忽略）"},
                    "preferences": {"type": "object", "description": "用户偏好，按 key 覆盖"},
                    "constraints": {"type": "array", "items": {"type": "string"}, "description": "约束条件，追加去重"},
                    "facts": {"type": "array", "items": {"type": "string"}, "description": "用户事实，追加"},
                    "feedback": {"type": "array", "items": {"type": "string"}, "description": "用户反馈，追加"},
                    "style_prefs": {"type": "object", "description": "风格偏好，按 key 覆盖"},
                },
            },
            func=update_profile,
        )

    def _commit_profile(self):
        if self._profile is not None and self._profile.dirty:
            try:
                self._profile.save(self.profile_dir)
            except Exception as e:
                logger.debug("档案卡保存失败: %s", e)

    def _retrieve_facts(self, task: str) -> str:
        if not self.facts_memory:
            return ""
        try:
            facts = self.facts_memory.search(task, top_k=FACTS_TOP_K)
        except Exception as e:
            logger.debug("facts 召回失败: %s", e)
            return ""
        if not facts:
            return ""
        return "\n".join(f"- {f}" for f in facts)

    def _extract_facts_from_conversation(self, messages):
        conversation = "\n".join(
            f"{m['role']}: {m.get('content', '')[:TRUNC_SHORT]}"
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        )
        if not conversation.strip():
            return []
        prompt = (
            "从以下对话中提取关于用户本人的长期事实（如职业、技能、偏好、背景、约束等）。"
            "只提取明确陈述的事实，不要推测，不要提取任务内容本身。"
            "只返回一个 JSON 数组，每个元素是一条简洁事实字符串；没有值得记住的事实则返回空数组 []。"
            "不要返回 JSON 之外的任何内容。\n\n" + conversation
        )
        extract_msgs = [
            {"role": "system", "content": "你是善于从对话中提取用户档案信息的助手。"},
            {"role": "user", "content": prompt},
        ]
        resp = self.model.generate(extract_msgs)
        match = re.search(r"\[.*\]", resp.content or "", re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except Exception as e:
            logger.debug("facts 提炼结果解析失败: %s", e)
            return []
        if not isinstance(parsed, list):
            return []
        facts = []
        for item in parsed:
            if isinstance(item, str):
                f = item.strip()
                if 2 <= len(f) <= TRUNC_MEDIUM:
                    facts.append(f)
        return facts

    def _auto_extract_facts(self, task, messages):
        if not self.auto_extract_facts or not self.facts_memory:
            return
        try:
            facts = self._extract_facts_from_conversation(messages)
        except Exception as e:
            logger.debug("facts 自动提炼失败: %s", e)
            return
        for f in facts:
            try:
                self.facts_memory.add(f)
            except Exception as e:
                logger.debug("fact 写入失败: %s", e)

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
        """滑动窗口：system + 最近 (window_size - 1) 条原文。"""
        if len(messages) <= self.window_size:
            return messages
        return [messages[0]] + messages[-(self.window_size - 1):]

    def _summarize_messages(self, messages):
        parts = []
        for msg in messages:
            if msg["role"] == "tool" and msg.get("content"):
                parts.append(f"工具返回: {msg['content'][:TRUNC_LONG]}")
            elif msg["role"] == "assistant" and msg.get("content"):
                parts.append(f"分析: {msg['content'][:TRUNC_MEDIUM]}")

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
            return resp.content or ""
        except Exception as e:
            logger.debug("LLM 摘要失败: %s", e)
            return "\n".join(parts[:3]) + "\n\n(超时，以上为部分结果)"

    def _retrieve_memory(self, task: str) -> list[MemoryHit]:
        if not self.memory:
            return []
        try:
            return self.memory.search(task, top_k=MEMORY_TOP_K)
        except Exception as e:
            logger.debug("记忆检索失败: %s", e)
            return []

    def _recall_text(self, task: str) -> str:
        """L5 召回层文本（MVP 保留现有向量召回）。"""
        history = self._retrieve_memory(task)
        if not history:
            return ""
        return "\n".join(f"- {h.task} → {h.document[:TRUNC_SHORT]}" for h in history)

    def _update_rolling_summary(self, overflow):
        """将新滑出窗口的消息增量摘要，追加到摘要块（L3）。"""
        if not overflow:
            return
        new_messages = overflow[self._summarized_upto:]
        if not new_messages:
            return
        new_summary = self._summarize_messages(new_messages)
        if new_summary:
            self.summary_block = self.summary_block + "\n\n" + new_summary if self.summary_block else new_summary
        self._summarized_upto = len(overflow)

    def _build_context(self, messages):
        """滑动窗口 + 滚动摘要 + 分层组装（L3/L4）。"""
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            rest = messages[1:]
        else:
            system_msg = {"role": "system", "content": self.system_prompt}
            rest = messages

        capacity = self.window_size - 1
        if len(rest) > capacity:
            window = rest[-capacity:]
            overflow = rest[:-capacity]
        else:
            window = rest
            overflow = []

        self._update_rolling_summary(overflow)

        return self.composer.compose(
            system_prompt=system_msg.get("content") or self.system_prompt,
            profile=self._profile.to_text() if self._profile else "",
            facts=getattr(self, "_facts", ""),
            recall=getattr(self, "_recall", ""),
            summary=self.summary_block,
            window=window,
        )

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
            if tn in self.tools and tn not in ("create_sub_agent", "final_answer", "update_profile"):
                sub_tools.append(self.tools[tn])
        if not sub_tools:
            sub_tools = [
                t for t in self.tools.values()
                if t.name not in ("create_sub_agent", "final_answer", "update_profile")
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
                max_steps=min(SUB_AGENT_MAX_STEPS, self.max_steps),
                window_size=self.window_size,
                allow_delegation=False,
                registry=registry,
            )
            registry.register(agent)
        return agent, task, None

    def _save_checkpoint(self, messages=None):
        if self.checkpoint and self.session_id:
            try:
                turn = getattr(self, "_current_turn", None)
                existing = self.checkpoint.load_full(self.session_id) or {}
                turns = list(existing.get("turns", []))
                if turn and turn.get("events"):
                    # 同一轮重复调用时不重复追加（对象同一性判断，进程内可靠）
                    if not turns or turns[-1] is not turn:
                        turns.append(turn)
                self.checkpoint.save(
                    self.session_id,
                    messages if messages is not None else self._last_messages,
                    turns=turns,
                    summary=self.summary_block,
                    summarized_upto=self._summarized_upto,
                )
            except Exception as e:
                logger.debug("检查点保存失败: %s", e)

    def _generate_stream(self, messages, tools_schema, delegation_id=None):
        """流式调用 LLM。yield token 事件，最终返回 ModelResponse。

        - 有 generate_stream 时：逐 token yield {"type":"token","content":...}
        - 无流式接口（测试 FakeModel）时：一次性 yield 完整 thought
        """
        if not hasattr(self.model, "generate_stream"):
            resp = self.model.generate(messages, tools_schema)
            if resp.content:
                ev = self._event("thought", content=resp.content)
                if delegation_id:
                    ev["delegation_id"] = delegation_id
                yield ev
            return resp

        text = ""
        tool_calls = None
        for chunk in self.model.generate_stream(messages, tools_schema):
            if chunk.content:
                text += chunk.content
                ev = self._event("token", content=chunk.content)
                if delegation_id:
                    ev["delegation_id"] = delegation_id
                yield ev
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls

        return ModelResponse(content=text or None, tool_calls=tool_calls)

    def run_stream(self, task: str, delegation_id: str | None = None):
        """核心执行逻辑的 generator。只 yield 事件，不打印，并收集本轮事件到 _current_turn。"""
        self._session_meta = SessionMetadata(
            session_id=self.session_id,
            agent_name=self.name,
            model=getattr(self.model, "model_id", None) or type(self.model).__name__,
        )
        try:
            turn = {"user": task, "events": []}
            self._current_turn = turn
            for ev in self._run_stream_inner(task, delegation_id):
                turn["events"].append(ev)
                yield ev
        finally:
            self._session_meta = None

    def _run_stream_inner(self, task: str, delegation_id: str | None = None):
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
            yield self._event("memory", hits=[asdict(h) for h in history])
        self._recall = self._recall_text(task)
        self._facts = self._retrieve_facts(task)
        self._profile = Profile.load(self.profile_dir) if self.profile_dir else None
        self.summary_block = ""
        self._summarized_upto = 0

        saved = None
        if self.checkpoint and self.session_id:
            saved = self.checkpoint.load(self.session_id)
            full = self.checkpoint.load_full(self.session_id) or {}
            self.summary_block = full.get("summary", "")
            self._summarized_upto = full.get("summarized_upto", 0)
        if saved:
            messages = list(saved)
            if messages and messages[0].get("role") == "system":
                messages[0] = {"role": "system", "content": self.system_prompt}
            else:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            messages = messages + [{"role": "user", "content": task}]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task},
            ]
        self._last_messages = messages

        idle_steps = 0
        for step in range(1, self.max_steps + 1):
            yield self._event("step", step=step, max_steps=self.max_steps)

            context = self._build_context(messages)
            tools_schema = self._build_tools_schema()
            msg = yield from self._generate_stream(context, tools_schema, delegation_id=delegation_id)
            text = msg.content or ""

            if not msg.tool_calls:
                idle_steps += 1
                messages.append({"role": "assistant", "content": text})
                # ponytail: 无工具调用 = 模型已给出直接回答，立即结束，避免重复生成
                if idle_steps >= 1:
                    note = f"[{self.name}] 连续多步未调用工具，直接结束。"
                    final = text or self._summarize_messages(messages)
                    yield self._event("note", content=note)
                    yield self._event("done", content=final, stored=False, session_id=self.session_id)
                    self._commit_profile()
                    self._save_checkpoint(messages)
                    return
                continue
            idle_steps = 0

            tool_calls_dict = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
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
                tool_name = tc.name
                args = tc.arguments

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
                for attempt in range(1, TOOL_RETRY_ATTEMPTS + 1):
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
                    if self.memory and self.store_policy.should_store(task, final):
                        self.memory.add(task, final)
                        stored = True
                    yield self._event("done", content=final, stored=stored, session_id=self.session_id)
                    self._commit_profile()
                    self._auto_extract_facts(task, messages)
                    self._save_checkpoint(messages)
                    return

        summary = self._summarize_messages(self._last_messages)
        yield self._event("note", content=f"[{self.name}] 达到最大步数，正在总结已有结果...")
        yield self._event("done", content=summary, stored=False, session_id=self.session_id)
        self._save_checkpoint(messages)

    def run(self, task: str) -> str:
        result = ""
        for event in self.run_stream(task):
            self._print_event(event)
            if event["type"] == "done" and "delegation_id" not in event:
                result = event["content"]
        return result

    def _print_event(self, event: dict):
        print_event(self.console, event, self.name, self.stream)
