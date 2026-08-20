"""Bing 搜索后端解析与兜底链测试。"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mini_smolagents.default_tools as _dt
from mini_smolagents import web_search
from mini_smolagents.default_tools import _parse_bing_html

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _clean_cache():
    _dt._search_cache.clear()
    yield
    _dt._search_cache.clear()


def _bing_results():
    return (FIXTURES / "bing_results.html").read_text(encoding="utf-8")


def test_parse_bing_extracts_all_results():
    items = _parse_bing_html(_bing_results())
    assert len(items) == 10
    assert all(i["title"] for i in items)
    assert all(i["href"].startswith("http") for i in items)
    assert all(len(i["body"]) >= 5 for i in items)


def test_parse_bing_empty_page():
    assert _parse_bing_html("<html></html>") == []


def test_bing_search_called_after_baidu_fails():
    items = [{"title": "Bing 结果", "href": "https://bing.example/x", "body": "内容"}]
    with patch("mini_smolagents.default_tools._baidu_search", side_effect=RuntimeError("拦截")) as bs, \
         patch("mini_smolagents.default_tools._bing_search", return_value=items) as bg, \
         patch("ddgs.DDGS") as ddgs_cls:
        out = web_search.func("广州天气")
    assert "Bing 结果" in out
    bs.assert_called_once()
    bg.assert_called_once()
    ddgs_cls.assert_not_called()


def test_ddgs_called_when_baidu_and_bing_fail():
    with patch("mini_smolagents.default_tools._baidu_search", side_effect=RuntimeError("拦截")), \
         patch("mini_smolagents.default_tools._bing_search", side_effect=ConnectionError("refused")), \
         patch("ddgs.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.text.return_value = [
            {"title": "DDG", "href": "https://ddg.example/x", "body": "fallback"},
        ]
        out = web_search.func("q")
    assert "fallback" in out
    ddgs_cls.assert_called_once()


def test_bing_throttle_sleeps():
    _dt._last_bing_ts = 1000.0
    with patch.object(_dt._time, "sleep") as sleeper, \
         patch("mini_smolagents.default_tools._bing_search", return_value=[{"title": "t", "href": "u", "body": "b"}]):
        web_search.func("q1")
        web_search.func("q2")
    assert sleeper.call_count >= 1