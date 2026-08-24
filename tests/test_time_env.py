"""环境感知测试：get_current_time 工具、搜索结果发布时间标注、reasoner CoT 透传。"""

import re
from types import SimpleNamespace

import pytest

from mini_smolagents import Agent, web_search
from mini_smolagents.types import ModelResponse, ToolCall
from backend.agents_config import build_agents


@pytest.fixture(scope="module")
def agents():
    return build_agents()


@pytest.fixture(autouse=True)
def _clean_cache():
    import mini_smolagents.default_tools as dt
    dt._search_cache.clear()
    yield
    dt._search_cache.clear()


def test_get_current_time_tool():
    from mini_smolagents import get_current_time
    out = get_current_time.func()
    assert re.match(r"\d{4}-\d{2}-\d{2} 周[一二三四五六日] \d{2}:\d{2}:\d{2}", out)


def test_get_current_time_is_tool():
    from mini_smolagents import get_current_time
    assert get_current_time.name == "get_current_time"
    assert "时间" in get_current_time.description
    assert get_current_time.parameters == {"type": "object", "properties": {}, "required": []}


def test_extract_pub_date_formats():
    from mini_smolagents.default_tools import _extract_pub_date
    assert _extract_pub_date("2026年8月20日 16:32 · 来源：东方财富") == "2026年8月20日"
    assert _extract_pub_date("2026-08-20 收盘 比亚迪") == "2026年8月20日"
    assert _extract_pub_date("2026/08/20 报告") == "2026年8月20日"
    assert _extract_pub_date("8月20日收盘价突破新高") == "8月20日"
    assert _extract_pub_date("没有任何日期信息的摘要") is None


def test_web_search_output_marks_pub_date():
    import mini_smolagents.default_tools as dt
    items = [{"title": "比亚迪股价走势", "href": "https://x/1", "body": "2026年8月20日 收盘报 300 元"}]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dt, "_baidu_search", lambda q, n: items)
        out = web_search.func("比亚迪 股价")
    assert "[发布于 2026年8月20日]" in out


def test_web_search_output_no_date_no_mark():
    import mini_smolagents.default_tools as dt
    items = [{"title": "比亚迪股价走势", "href": "https://x/1", "body": "今日收盘上涨"}]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dt, "_baidu_search", lambda q, n: items)
        out = web_search.func("比亚迪 股价")
    assert "[发布于" not in out


def test_llm_reasoner_flag():
    from mini_smolagents.llm import OpenAIModel
    assert OpenAIModel("deepseek-reasoner", api_key="k")._is_reasoner is True
    assert OpenAIModel("deepseek-chat", api_key="k")._is_reasoner is False


def test_llm_request_kwargs_filters_temperature():
    from mini_smolagents.llm import OpenAIModel
    r = OpenAIModel("deepseek-reasoner", api_key="k", temperature=0.0, top_p=0.8)
    assert "temperature" not in r._request_kwargs()
    assert "top_p" not in r._request_kwargs()
    c = OpenAIModel("deepseek-chat", api_key="k", temperature=0.0)
    assert c._request_kwargs()["temperature"] == 0.0


def test_llm_to_response_parses_reasoning():
    from mini_smolagents.llm import OpenAIModel
    msg = SimpleNamespace(
        content="回答",
        tool_calls=None,
        reasoning_content="先判断时间，再决定搜索词",
    )
    resp = OpenAIModel("deepseek-reasoner", api_key="k")._to_response(msg)
    assert resp.content == "回答"
    assert resp.reasoning == "先判断时间，再决定搜索词"


def test_agents_use_reasoner_and_time_tool(agents):
    main, pm, developer, reviewer, researcher = (
        agents["助手"], agents["PM"], agents["developer"], agents["reviewer"], agents["researcher"]
    )
    assert main.model.model_id == "deepseek-reasoner"
    assert researcher.model.model_id == "deepseek-reasoner"
    assert pm.model.model_id == "deepseek-chat"
    assert developer.model.model_id == "deepseek-chat"
    assert reviewer.model.model_id == "deepseek-chat"
    for a in (main, pm, developer, reviewer, researcher):
        assert "get_current_time" in a.tools


def test_generate_stream_reasoning_non_stream():
    class Fake:
        def generate(self, messages, tools=None):
            return ModelResponse(content="最终回答", reasoning="这是推理过程")

    agent = Agent(model=Fake(), tools=[])
    events = list(agent._generate_stream([{"role": "user", "content": "x"}], None))
    thoughts = [e for e in events if e["type"] == "thought"]
    assert any(e.get("reasoning") and e["content"] == "这是推理过程" for e in thoughts)
    assert any(not e.get("reasoning") and e["content"] == "最终回答" for e in thoughts)


def test_generate_stream_reasoning_streaming():
    class Fake:
        def generate_stream(self, messages, tools=None):
            yield ModelResponse(reasoning="先")
            yield ModelResponse(reasoning="判断")
            yield ModelResponse(content="答")

    agent = Agent(model=Fake(), tools=[])
    events = list(agent._generate_stream([{"role": "user", "content": "x"}], None))
    thoughts = [e["content"] for e in events if e["type"] == "thought" and e.get("reasoning")]
    assert "".join(thoughts) == "先判断"
    assert any(e["type"] == "token" and e["content"] == "答" for e in events)
