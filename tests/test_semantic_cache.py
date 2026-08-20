"""web_search 语义去重缓存测试（真实 bge embedding，标定过的 query 对）。"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mini_smolagents.default_tools as _dt
from mini_smolagents import web_search

BASE = "比亚迪 股价 今日 2025"
VARIANTS = [
    "比亚迪 002594 股价 今日收盘",
    "比亚迪 01211 港股 股价 今日",
    "比亚迪 002594 股价 东方财富 今日",
    "比亚迪股份 01211 港股 股价 今日 收盘",
]
ITEMS = [{"title": "缓存结果", "href": "https://x/1", "body": "来自首个查询的结果"}]


@pytest.fixture(autouse=True)
def _clean_cache():
    _dt._search_cache.clear()
    yield
    _dt._search_cache.clear()


def _patch_backend():
    return patch("mini_smolagents.default_tools._baidu_search", return_value=ITEMS)


def test_variant_served_from_semantic_cache():
    with _patch_backend() as bs:
        out1 = web_search.func(BASE)
        out2 = web_search.func(VARIANTS[0])
    assert out1 == out2
    assert bs.call_count == 1


def test_all_calibrated_variants_hit():
    with _patch_backend() as bs:
        web_search.func(BASE)
        for v in VARIANTS:
            web_search.func(v)
    assert bs.call_count == 1


def test_no_shared_token_breaks_semantic_match():
    with _patch_backend() as bs:
        web_search.func(BASE)
        out2 = web_search.func("特斯拉 Tesla 股价 今日 2025")
    assert bs.call_count == 2
    assert "缓存结果" in out2


def test_below_threshold_breaks_semantic_match():
    with _patch_backend() as bs:
        web_search.func(BASE)
        web_search.func("比亚迪 2025 年全年营业收入 财报")
    assert bs.call_count == 2


def test_embedding_unavailable_falls_to_exact_only():
    with patch("mini_smolagents.default_tools.get_embedding_function", return_value=None), \
         _patch_backend() as bs:
        web_search.func(BASE)
        web_search.func(BASE)
        web_search.func(VARIANTS[0])
    assert bs.call_count == 2


def test_log_marks_semantic_dup_hit():
    with _patch_backend(), patch("mini_smolagents.default_tools._log_search") as lg:
        web_search.func(BASE)
        web_search.func(VARIANTS[0])
    assert lg.call_count == 2
    assert lg.call_args_list[0].args[0]["dup_hit"] is None
    assert lg.call_args_list[1].args[0]["dup_hit"] == "semantic"


def test_exact_match_still_works():
    with _patch_backend() as bs:
        web_search.func(BASE)
        web_search.func(BASE)
    assert bs.call_count == 1


def test_two_char_entity_variants_hit():
    cases = [
        ("广州 天气 今日", "广州 明天 天气 预报"),
        ("北京 房价 2026", "北京 新房 房价走势"),
    ]
    for first, second in cases:
        _dt._search_cache.clear()
        with _patch_backend() as bs:
            out1 = web_search.func(first)
            out2 = web_search.func(second)
        assert bs.call_count == 1, f"{first!r} -> {second!r} 未命中语义缓存"
        assert out1 == out2


def test_two_char_entity_different_topic_blocked():
    with _patch_backend() as bs:
        web_search.func("广州 天气")
        out2 = web_search.func("广州 美食推荐")
    assert bs.call_count == 2
    assert "缓存结果" in out2