import html as _html
import http.cookiejar as _cookiejar
import io
import re as _re
import urllib.parse as _urlparse
import urllib.request as _urlrequest
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
import json as _json
import threading as _threading
import time as _time
from pathlib import Path as _Path

from ._exec import run_with_timeout
from .embedding import get_embedding_function
from .config import BAIDU_MIN_INTERVAL, BAIDU_PAGE_TIMEOUT, BING_MIN_INTERVAL, BING_PAGE_TIMEOUT, PYTHON_INTERPRETER_TIMEOUT, SEARCH_CACHE_SIZE, SEARCH_CACHE_TTL, SEARCH_SEMANTIC_DUP_THRESHOLD, TRUNC_SHORT, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT
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

_SEARCH_LOG_PATH = _Path(__file__).resolve().parent.parent / "search_log.jsonl"
_search_log_lock = _threading.Lock()


def _log_search(entry: dict):
    try:
        with _search_log_lock, open(_SEARCH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


_baidu_opener = None

_search_cache = OrderedDict()
_last_baidu_ts = 0.0


_token_pat = _re.compile(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{3,}|[0-9]{5,}")


def _shared_key_tokens(a: str, b: str) -> bool:
    def grams(text):
        out = set()
        for m in _token_pat.findall(text):
            if any("\u4e00" <= ch <= "\u9fff" for ch in m):
                for i in range(len(m) - 2):
                    out.add(m[i:i + 3])
            else:
                out.add(m)
        return out
    return bool(grams(a) & grams(b))


def _emb_texts(texts) -> list | None:
    fn = get_embedding_function()
    if fn is None:
        return None
    try:
        return fn(texts)
    except Exception:
        return None


def _emb_query(text) -> list | None:
    embs = _emb_texts([text])
    return embs[0] if embs else None


def _cos_sim(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cache_get(key):
    now = _time.monotonic()
    hit = _search_cache.pop(key, None)
    if hit is not None:
        if now - hit["ts"] <= SEARCH_CACHE_TTL:
            _search_cache[key] = hit
            return hit["value"]
        _search_cache.pop(key, None)
    return None


def _cache_lookup(query, max_results, emb):
    """精确匹配优先；语义匹配（cos>=阈值 且 共享关键 token）返回缓存结果。"""
    exact = None
    for k in list(_search_cache):
        if k[0] == query and k[1] == max_results:
            exact = k
            break
    if exact is not None:
        hit = _search_cache.pop(exact, None)
        if hit is not None and _time.monotonic() - hit["ts"] <= SEARCH_CACHE_TTL:
            _search_cache[exact] = hit
            return hit["value"], "exact"
        _search_cache.pop(exact, None)
    if emb is None:
        return None, None
    now = _time.monotonic()
    for k, v in list(_search_cache.items()):
        if k[1] != max_results:
            continue
        cemb = v.get("emb")
        if cemb is None or now - v["ts"] > SEARCH_CACHE_TTL:
            continue
        if _cos_sim(emb, cemb) >= SEARCH_SEMANTIC_DUP_THRESHOLD and _shared_key_tokens(query, k[0]):
            v = _search_cache.pop(k, None)
            if v is not None:
                v["ts"] = now
                _search_cache[k] = v
                return v["value"], "semantic"
    return None, None


def _cache_put(key, value, emb=None):
    _search_cache.pop(key, None)
    _search_cache[key] = {"ts": _time.monotonic(), "value": value, "emb": emb}
    while len(_search_cache) > SEARCH_CACHE_SIZE:
        _search_cache.popitem(last=False)


def _baidu_throttle():
    global _last_baidu_ts
    wait = BAIDU_MIN_INTERVAL - (_time.monotonic() - _last_baidu_ts)
    if wait > 0:
        _time.sleep(wait)
    _last_baidu_ts = _time.monotonic()


_bing_opener = None
_last_bing_ts = 0.0


def _get_bing_opener():
    global _bing_opener
    if _bing_opener is None:
        cj = _cookiejar.CookieJar()
        opener = _urlrequest.build_opener(_urlrequest.HTTPCookieProcessor(cj))
        opener.addheaders = list(_BAIDU_HEADERS.items())
        _bing_opener = opener
    return _bing_opener


def _bing_throttle():
    global _last_bing_ts
    wait = BING_MIN_INTERVAL - (_time.monotonic() - _last_bing_ts)
    if wait > 0:
        _time.sleep(wait)
    _last_bing_ts = _time.monotonic()


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


def _parse_bing_html(html_text: str) -> list[dict]:
    """解析必应搜索结果页：b_algo 容器内提取标题链接（h2>a）与摘要（b_caption）。"""
    results = []
    for block in _re.split(r'<li class="b_algo"', html_text)[1:]:
        a = _re.search(r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, _re.DOTALL)
        if not a:
            continue
        href = a.group(1)
        title = _html.unescape(_re.sub(r"<[^>]+>", "", a.group(2))).strip()
        if not title:
            continue
        if not href.startswith("http"):
            href = "https://cn.bing.com" + href
        snippet = ""
        cap = _re.search(r'<div class="b_caption">(.*?)</div>', block, _re.DOTALL)
        if cap:
            snippet = _html.unescape(_re.sub(r"<[^>]+>", "", cap.group(1))).strip()[:TRUNC_SHORT]
        results.append({"title": title, "href": href, "body": snippet})
    return results


def _bing_search(query: str, max_results: int) -> list[dict]:
    """必应（cn.bing.com）网页搜索。"""
    opener = _get_bing_opener()
    url = "https://cn.bing.com/search?" + _urlparse.urlencode({"q": query, "count": max_results})
    with opener.open(url, timeout=BING_PAGE_TIMEOUT) as resp:
        html_text = resp.read().decode("utf-8", errors="ignore")
    return _parse_bing_html(html_text)


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
    """Search the web for the given query. Returns up to 10 results with titles, URLs, and snippets. Baidu primary, Bing fallback, DuckDuckGo last. \u540c\u4e00\u4e3b\u9898\u53ea\u641c\u7d22\u4e00\u6b21\uff0c\u4e00\u6b21\u67e5\u8be2\u8981\u5b8c\u6574\uff08\u4e3b\u9898+\u8303\u56f4+\u9650\u5b9a\u8bcd\uff09\uff0c\u4e0d\u8981\u7528\u53d8\u4f53 query \u53cd\u590d\u641c\u7d22\u540c\u4e00\u4e3b\u9898\u3002"""
    result: dict = {"output": "", "error": "", "timed_out": False}

    def _format(items):
        return "\n\n".join(
            f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}" for i, r in enumerate(items, 1)
        )

    _t0 = _time.monotonic()
    _source = "error"
    _dup_hit = None
    _query_emb = _emb_query(query)

    def _do_search():
        nonlocal _source, _dup_hit
        cache_key = (query, WEB_SEARCH_MAX_RESULTS)
        cached, _dup_hit = _cache_lookup(query, WEB_SEARCH_MAX_RESULTS, _query_emb)
        if cached is not None:
            _source = "cache"
            result["output"] = cached
            return

        errors = []
        try:
            _baidu_throttle()
            items = _baidu_search(query, WEB_SEARCH_MAX_RESULTS)
            if items:
                _source = "baidu"
                output = _format(items)
                _cache_put(cache_key, output, emb=_query_emb)
                result["output"] = output
                return
            errors.append("baidu: no results")
        except Exception as e:
            errors.append(f"baidu: {type(e).__name__}: {e}")

        try:
            _bing_throttle()
            items = _bing_search(query, WEB_SEARCH_MAX_RESULTS)
            if items:
                _source = "bing"
                output = _format(items)
                _cache_put(cache_key, output, emb=_query_emb)
                result["output"] = output
                return
            errors.append("bing: no results")
        except Exception as e:
            errors.append(f"bing: {type(e).__name__}: {e}")

        try:
            from ddgs import DDGS
            with DDGS(timeout=8) as ddgs:
                items = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS))
            if items:
                output = _format(items)
                _cache_put(cache_key, output)
                result["output"] = output
                return
            errors.append("ddgs: no results")
        except Exception as e:
            errors.append(f"ddgs: {type(e).__name__}: {e}")

        result["error"] = "; ".join(errors)

    result["timed_out"] = run_with_timeout(_do_search, WEB_SEARCH_TIMEOUT)

    _log_search({
        "ts": round(_time.time(), 3),
        "query": query,
        "source": _source,
        "dup_hit": _dup_hit,
        "elapsed_ms": int((_time.monotonic() - _t0) * 1000),
        "timed_out": result["timed_out"],
        "error": result["error"] or None,
    })

    if result["timed_out"]:
        return f"Error: web search timed out. 全部后端失败，{WEB_SEARCH_TIMEOUT} 秒预算耗尽。\n        百度风控拦截时需 IP 冷却，请稍后再试或更换网络。"
    if result["error"]:
        if "安全验证" in result["error"] or "拦截" in result["error"]:
            return "Error: 百度风控拦截（IP 冷却中，通常数小时），请稍后再试或更换网络。"
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
