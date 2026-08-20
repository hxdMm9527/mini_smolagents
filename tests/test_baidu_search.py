"""百度搜索解析与 web_search 兜底链测试。"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mini_smolagents.default_tools as _dt
from mini_smolagents.default_tools import (
    _is_baidu_verification,
    _parse_baidu_html,
    web_search,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _clean_cache():
    _dt._search_cache.clear()
    yield
    _dt._search_cache.clear()


def _load(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_baidu_extracts_titles_links():
    items = _parse_baidu_html(_load("baidu_results_sample.html"))
    assert len(items) == 3
    assert items[0]["title"].startswith("广州天气_广州市天气预报")
    assert items[0]["href"].startswith("http")
    assert "中国天气网" in items[0]["title"]


def test_parse_baidu_skips_slots_and_dedups():
    items = _parse_baidu_html(_load("baidu_results_sample.html"))
    urls = [i["href"] for i in items]
    assert len(urls) == len(set(urls))
    assert "腾讯天气" in items[2]["title"]
    assert items[2]["href"] != items[1]["href"]
    assert "重复链接" not in "".join(i["title"] for i in items)


def test_parse_baidu_snippet():
    items = _parse_baidu_html(_load("baidu_results_sample.html"))
    assert len(items[0]["body"]) >= 15


def test_parse_baidu_empty_page():
    assert _parse_baidu_html("<html><body>no results</body></html>") == []


def test_verification_page_detection():
    assert _is_baidu_verification(_load("baidu_verify.html")) is True
    assert _is_baidu_verification(_load("baidu_results_sample.html")) is False


def test_web_search_uses_baidu_and_skips_ddgs():
    with patch("mini_smolagents.default_tools._baidu_search") as bs:
        bs.return_value = [{"title": "广州天气", "href": "https://example.com/gz", "body": "今天多云"}]
        out = web_search.func("广州天气")
    assert "广州天气" in out
    assert "example.com" in out
    bs.assert_called_once()
    assert "ddgs" not in out.lower()


def test_web_search_falls_back_to_ddgs():
    with patch("mini_smolagents.default_tools._baidu_search", side_effect=RuntimeError("拦截")), \
         patch("ddgs.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.text.return_value = [
            {"title": "DuckDuckGo 结果", "href": "https://ddg.example/x", "body": "fallback 内容"},
        ]
        out = web_search.func("query")
    assert "fallback 内容" in out
    assert "Error" not in out


def test_web_search_both_fail_reports_error():
    with patch("mini_smolagents.default_tools._baidu_search", side_effect=ConnectionError("refused")), \
         patch("ddgs.DDGS", side_effect=Exception("timeout")):
        out = web_search.func("query")
    assert out.startswith("Error:")
    assert "baidu" in out
    assert "ddgs" in out