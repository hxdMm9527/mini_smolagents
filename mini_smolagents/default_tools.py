import io
import threading
from contextlib import redirect_stderr, redirect_stdout

from .tools import tool

ALLOWED_IMPORTS = ["math", "json", "re", "datetime", "random", "collections"]
ALLOWED_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "dir": dir, "enumerate": enumerate, "filter": filter, "float": float,
    "getattr": getattr, "int": int, "isinstance": isinstance, "len": len,
    "list": list, "map": map, "max": max, "min": min,
    "print": print, "range": range, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type,
    "zip": zip, "True": True, "False": False, "None": None,
    "Exception": Exception, "TypeError": TypeError, "ValueError": ValueError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
}

_real_import = __import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed. Allowed imports: {ALLOWED_IMPORTS}")
    return _real_import(name, globals, locals, fromlist, level)


@tool
def final_answer(answer) -> str:
    """Return the final answer to the task. Call this when you have completed the task."""
    return str(answer)


@tool
def web_search(query: str) -> str:
    """Search the web for the given query. Returns up to 10 results with titles, URLs, and snippets."""
    from ddgs import DDGS

    result: dict = {"output": "", "error": "", "timed_out": False}

    def _do_search():
        try:
            with DDGS() as ddgs:
                items = list(ddgs.text(query, max_results=10))

            if not items:
                result["output"] = "No results found."
                return

            lines = []
            for i, r in enumerate(items, 1):
                lines.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}")
            result["output"] = "\n\n".join(lines)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_do_search, daemon=True)
    t.start()
    t.join(timeout=15)
    if t.is_alive():
        result["timed_out"] = True

    if result["timed_out"]:
        return "Error: web search timed out (15-second limit)."
    if result["error"]:
        return f"Error: {result['error']}"
    return result["output"]


@tool
def python_interpreter(code: str) -> str:
    """Execute Python code and return the output. For math, calculations, data processing, and string manipulation.
    Supports imports: math, json, re, datetime, random, collections.
    Has a 10-second timeout. Use print() to output results for multi-line code."""
    restricted_globals = {
        "__builtins__": {**ALLOWED_BUILTINS, "__import__": _safe_import},
    }

    result: dict = {"output": "", "error": "", "timed_out": False}

    def _run():
        f = io.StringIO()
        try:
            with redirect_stdout(f), redirect_stderr(f):
                try:
                    value = str(eval(code, restricted_globals))
                    stdout = f.getvalue().strip()
                    result["output"] = (stdout + "\n" + value).strip() if stdout else value
                except SyntaxError:
                    exec(code, restricted_globals)
                    result["output"] = f.getvalue().strip()
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=10)
    if t.is_alive():
        result["timed_out"] = True

    if result["timed_out"]:
        return "Error: code execution timed out (10-second limit)."
    if result["error"]:
        return f"Error: {result['error']}"
    return result["output"] if result["output"] else "(no output)"
