import io
import re
import time
from contextlib import redirect_stderr, redirect_stdout

from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from ._exec import run_with_timeout
from .config import CODE_EXEC_TIMEOUT, CODE_MAX_STEPS, DEFAULT_WINDOW_SIZE, TRUNC_MEDIUM
from .agent import Agent
from .console import _trunc
from .default_tools import ALLOWED_BUILTINS, ALLOWED_IMPORTS, _safe_import
from .prompts import CODE_SYSTEM_PROMPT


class _StopExec(BaseException):
    pass


def _extract_code(text: str) -> str:
    match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class CodeAgent(Agent):
    def __init__(self, model, tools, max_steps=CODE_MAX_STEPS, window_size=DEFAULT_WINDOW_SIZE, additional_imports=None, name=None, description=None, managed_agents=None):
        super().__init__(model, tools, max_steps, window_size, name=name, description=description, managed_agents=managed_agents)
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

        result["timed_out"] = run_with_timeout(_run, CODE_EXEC_TIMEOUT)

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
                resp = self.model.generate(self._get_trimmed_messages(messages))
                full_text = resp.content or ""
                if full_text:
                    with Live("", console=self.console, refresh_per_second=60) as live:
                        for i in range(1, len(full_text) + 1, 2):
                            live.update(Text(full_text[:i]))
                            time.sleep(0.01)
                        live.update(Text(full_text))
            else:
                resp = self.model.generate(self._get_trimmed_messages(messages))
                full_text = resp.content or ""
                if full_text:
                    self.console.print(Text(full_text[:TRUNC_MEDIUM]))

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
