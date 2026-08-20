"""web_search 缓存/限频/错误信息测试。"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mini_smolagents.default_tools as dt
from mini_smolagents import web_search

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _clean_cache():
    dt._search_cache.clear()
    dt._last_baidu_ts = 0.0
    yield
    dt._search_cache.clear()


def test_same_query_served_from_cache():
    items = [{"title": "a", "href": "https://x/a", "body": "内容"}]
    with patch("mini_smolagents.default_tools._baidu_search", return_value=items) as bs:
        out1 = web_search.func("广州天气")
        out2 = web_search.func("广州天气")
    assert out1 == out2
    assert bs.call_count == 1


def test_different_query_not_cached():
    items = [{"title": "a", "href": "https://x/a", "body": "内容"}]
    with patch("mini_smolagents.default_tools._baidu_search", return_value=items) as bs:
        web_search.func("广州天气")
        web_search.func("深圳天气")
    assert bs.call_count == 2


def test_baidu_throttle_sleeps_between_calls():
    dt._last_baidu_ts = 1000.0
    with patch.object(dt._time, "sleep") as sleeper, \
         patch("mini_smolagents.default_tools._baidu_search", return_value=[{"title": "t", "href": "u", "body": "b"}]):
        web_search.func("q1")
        web_search.func("q2")
    assert sleeper.call_count >= 1


def test_error_message_mentions_cooldown():
    with patch("mini_smolagents.default_tools._baidu_search", side_effect=RuntimeError("百度安全验证拦截")), \
         patch("mini_smolagents.default_tools._bing_search", side_effect=ConnectionError("refused")), \
         patch("ddgs.DDGS", side_effect=Exception("timeout")):
        out = web_search.func("广州天气")
    assert "风控拦截" in out
    assert "冷却" in out


def test_cache_ttl_expiry():
    items = [{"title": "a", "href": "https://x/a", "body": "内容"}]
    with patch("mini_smolagents.default_tools._baidu_search", return_value=items) as bs:
        web_search.func("广州天气")
        dt._search_cache[(("广州天气", dt.WEB_SEARCH_MAX_RESULTS))]["ts"] -= (dt.SEARCH_CACHE_TTL + 10)
        web_search.func("广州天气")
    assert bs.call_count == 2