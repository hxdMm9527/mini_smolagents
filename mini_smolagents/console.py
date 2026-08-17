import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


def _trunc(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def print_event(console: Console, event: dict, name: str, stream: bool) -> None:
    etype = event["type"]
    who = event.get("agent", name)
    did = f" (sub:{event['delegation_id'][:8]})" if event.get("delegation_id") else ""
    if etype == "step":
        console.print(Rule(f"[{who}] Step {event['step']}/{event['max_steps']}{did}", style="bold blue"))
    elif etype == "thought":
        text = event["content"]
        if stream:
            with Live("", console=console, refresh_per_second=60) as live:
                for i in range(1, len(text) + 1, 2):
                    live.update(Markdown(text[:i]))
                    time.sleep(0.01)
                live.update(Markdown(text))
        else:
            console.print(Markdown(text))
    elif etype == "token":
        pass
    elif etype == "action":
        action_text = Text(f"[{who}] Action: {event['tool']}", style="bold yellow")
        action_text.append(f"\nArgs: {_trunc(str(event['args']), 200)}", style="dim")
        console.print(Panel(action_text, border_style="yellow"))
    elif etype == "result":
        console.print(Panel(Text(str(event["content"])[:500], style="green"), border_style="green", title=f"[{who}] Result"))
    elif etype == "note":
        console.print(Panel(Text(event["content"], style="orange3"), border_style="orange3"))
    elif etype == "done":
        console.print(Panel(Text(str(event["content"]), style="bold gold1"), border_style="gold1", title=f"[{who}] Done"))
