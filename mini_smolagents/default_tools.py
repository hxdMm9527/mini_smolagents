import html as _html
import http.cookiejar as _cookiejar
import io
import re as _re
import urllib.parse as _urlparse
import urllib.request as _urlrequest
from contextlib import redirect_stderr, redirect_stdout

from ._exec import run_with_timeout
from .config import BAIDU_PAGE_TIMEOUT, PYTHON_INTERPRETER_TIMEOUT, TRUNC_SHORT, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT
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


_BAIDU_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

_baidu_opener = None


def _get_baidu_opener():
    global _baidu_opener
    if _baidu_opener is None:
        cj = _cookiejar.CookieJar()
        opener = _urlrequest.build_opener(_urlrequest.HTTPCookieProcessor(cj))
        opener.addheaders = list(_BAIDU_HEADERS.items())
        try:
            opener.open("https://www.baidu.com", timeout=BAIDU_PAGE_TIMEOUT).read()
        except Exception:
            pass
        _baidu_opener = opener
    return _baidu_opener


def _parse_baidu_html(html_text: str) -> list[dict]:
    """解析百度搜索结果页：提取 h3 内标题链接 + 就近摘要文本。"""
    results = []
    seen = set()
    for m in _re.finditer(r"<h3[^>]*>(.*?)</h3>", html_text, _re.DOTALL):
        a = _re.search(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', m.group(1), _re.DOTALL)
        if not a:
            continue
        href = a.group(1)
        title = _html.unescape(_re.sub(r"<[^>]+>", "", a.group(2))).strip().rstrip(" -")
        if not title:
            continue
        if not href.startswith("http"):
            href = "https://www.baidu.com" + href
        if href in seen:
            continue
        seen.add(href)
        snippet = ""
        tail = _html.unescape(html_text[m.end():m.end() + 4000])
        for frag in _re.split(r"<[^>]+>", tail):
            frag = frag.strip()
            if len(frag) >= 15:
                snippet = frag[:TRUNC_SHORT]
                break
        results.append({"title": title, "href": href, "body": snippet})
    return results


def _is_baidu_verification(html_text: str) -> bool:
    return "安全验证" in html_text


def _baidu_search(query: str, max_results: int) -> list[dict]:
    """百度网页搜索（带 cookie 会话，反爬验证页快速失败）。"""
    opener = _get_baidu_opener()
    url = "https://www.baidu.com/s?" + _urlparse.urlencode({"wd": query, "rn": max_results})
    with opener.open(url, timeout=BAIDU_PAGE_TIMEOUT) as resp:
        html_text = resp.read().decode("utf-8", errors="ignore")
    if _is_baidu_verification(html_text):
        raise RuntimeError("百度安全验证拦截，稍后重试")
    return _parse_baidu_html(html_text)


@tool
def web_search(query: str) -> str:
    """Search the web for the given query. Returns up to 10 results with titles, URLs, and snippets. Baidu primary, DuckDuckGo fallback."""
    result: dict = {"output": "", "error": "", "timed_out": False}

    def _format(items):
        return "\n\n".join(
            f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}" for i, r in enumerate(items, 1)
        )

    def _do_search():
        errors = []
        try:
            items = _baidu_search(query, WEB_SEARCH_MAX_RESULTS)
            if items:
                result["output"] = _format(items)
                return
            errors.append("baidu: no results")
        except Exception as e:
            errors.append(f"baidu: {type(e).__name__}: {e}")

        try:
            from ddgs import DDGS
            with DDGS(timeout=8) as ddgs:
                items = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS))
            if items:
                result["output"] = _format(items)
                return
            errors.append("ddgs: no results")
        except Exception as e:
            errors.append(f"ddgs: {type(e).__name__}: {e}")

        result["error"] = "; ".join(errors)

    result["timed_out"] = run_with_timeout(_do_search, WEB_SEARCH_TIMEOUT)

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

    result["timed_out"] = run_with_timeout(_run, PYTHON_INTERPRETER_TIMEOUT)

    if result["timed_out"]:
        return "Error: code execution timed out (10-second limit)."
    if result["error"]:
        return f"Error: {result['error']}"
    return result["output"] if result["output"] else "(no output)"
