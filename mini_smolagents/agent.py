import io
import json
import re
import threading
import time
from contextlib import redirect_stderr, redirect_stdout

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .default_tools import ALLOWED_BUILTINS, ALLOWED_IMPORTS, _safe_import, final_answer as _FINAL_ANSWER_TOOL
from .types import Tool

SYSTEM_PROMPT = """\
你是一个善于逐步解决问题的助手。你可以使用工具调用来完成任务。
当你有最终答案时，调用 `final_answer` 工具。
不要在相同的参数下重复调用同一个工具。\
"""


class Agent:
    def __init__(self, model, tools, max_steps=10, max_messages=30, stream=False, name=None, description=None, managed_agents=None):
        self.model = model
        self.max_steps = max_steps
        self.max_messages = max_messages
        self.stream = stream
        self.console = Console()
        self.name = name
        self.description = description
        self.tools = {}

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

                def _make_managed_call(_sub):
                    def _call(task: str = "") -> str:
                        return _sub.run(task)
                    return _call

                self.tools[sub_name] = Tool(
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
                    func=_make_managed_call(sub),
                )

        if "final_answer" not in self.tools:
            self.tools["final_answer"] = _FINAL_ANSWER_TOOL

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
        return schema

    def _get_trimmed_messages(self, messages):
        if len(messages) <= self.max_messages:
            return messages
        return [messages[0], messages[1]] + messages[-(self.max_messages - 2):]

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        tools_schema = self._build_tools_schema()

        for step in range(1, self.max_steps + 1):
            self.console.print(Rule(f"Step {step}/{self.max_steps}", style="bold blue"))

            trimmed = self._get_trimmed_messages(messages)

            if self.stream:
                response = self.model.generate(trimmed, tools_schema)
                msg = response.choices[0].message
                text = msg.content or ""

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        text += f"\n\n🔧 **{tc.function.name}**"
                        args_str = tc.function.arguments
                        if isinstance(args_str, str):
                            text += f"\n```json\n{args_str}\n```"

                if text:
                    with Live("", console=self.console, refresh_per_second=60) as live:
                        for i in range(1, len(text) + 1, 2):
                            live.update(Markdown(text[:i]))
                            time.sleep(0.01)
                        live.update(Markdown(text))
            else:
                response = self.model.generate(trimmed, tools_schema)
                msg = response.choices[0].message
                if msg.content:
                    self.console.print(Markdown(msg.content))

            if not msg.tool_calls:
                text = msg.content or ""
                messages.append({"role": "assistant", "content": text})
                continue

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

                result = None
                for attempt in range(1, 4):
                    try:
                        result = self.tools[tool_name].func(**args)
                        break
                    except Exception as e:
                        if attempt == 3:
                            result = f"Error after 3 retries: {type(e).__name__}: {e}"

                action_text = Text(f"Action: {tool_name}", style="bold yellow")
                action_text.append(f"\nArgs: {_trunc(str(args), 200)}", style="dim")
                self.console.print(Panel(action_text, border_style="yellow"))

                result_text = Text(str(result)[:500], style="green")
                self.console.print(Panel(result_text, border_style="green", title="Result"))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

                if tool_name == "final_answer":
                    self.console.print(Panel(Text(str(result), style="bold gold1"), border_style="gold1", title="Done"))
                    return str(result)

        raise RuntimeError(
            f"Agent 达到最大步数（{self.max_steps}）但未完成任务。"
        )


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
            self.console.print(Rule(f"Step {step}/{self.max_steps}", style="bold blue"))

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
                self.console.print(Panel(Text(value_short, style="red"), border_style="red", title="Error"))
            elif status == "final_answer":
                self.console.print(Panel(Text(value_short, style="bold gold1"), border_style="gold1", title="Code output"))
            else:
                self.console.print(Panel(Text(value_short, style="green"), border_style="green", title="Code output"))
            messages.append({"role": "user", "content": value})

            if status == "final_answer":
                self.console.print(Panel(Text(value, style="bold gold1"), border_style="gold1", title="Done"))
                return value

        raise RuntimeError(
            f"Agent 达到最大步数（{self.max_steps}）但未完成任务。"
        )
