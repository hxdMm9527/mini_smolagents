"""主 Agent 搜索委派引导测试：描述重构（A）+ 渐进式引导（B）。"""

import pytest
from unittest.mock import patch

from mini_smolagents import Agent, web_search
from mini_smolagents.types import ModelResponse, ToolCall
from backend.agents_config import MAIN_PROMPT, build_agents

B = "比亚迪 股价 今日 2025"
B2 = "比亚迪 002594 股价 东方财富 今日"
B3 = "比亚迪 今天 股票 收盘价"
T = "特斯拉 Tesla 股价 今日 2025"


@pytest.fixture(scope="module")
def agents():
    return build_agents()


def _mk_search_tool():
    """复制一个独立 web_search Tool（func 为 mock，不污染模块级共享实例）。"""
    from mini_smolagents.types import Tool
    return Tool(
        name=web_search.name,
        description=web_search.description,
        parameters=web_search.parameters,
        func=lambda query: "mock-result",
    )


def _mk_agent(**kw):
    model = type("M", (), {"generate": lambda self, m, t=None: ModelResponse(content="ok")})()
    return Agent(model=model, tools=[_mk_search_tool()], search_delegate_hint=True, **kw)


def test_web_search_description_limits(agents):
    assert "单次查询工具" in web_search.description
    assert "researcher" in web_search.description
    assert "多主题对比" in web_search.description


def test_researcher_description_is_entrance(agents):
    researcher = agents["researcher"]
    assert "搜索主入口" in researcher.description
    assert "不占用你的上下文" in researcher.description
    assert "多轮" in researcher.description


def test_main_prompt_removes_escape_route(agents):
    assert "任何可能需要 2 次以上搜索的主题一律交给 researcher" in MAIN_PROMPT
    assert "不要自己用变体 query 反复搜索" not in MAIN_PROMPT


def test_main_enables_search_delegate_hint(agents):
    main, pm, developer, reviewer, researcher = (
        agents["助手"], agents["PM"], agents["developer"], agents["reviewer"], agents["researcher"]
    )
    assert main.search_delegate_hint is True
    assert pm.search_delegate_hint is False
    assert developer.search_delegate_hint is False
    assert reviewer.search_delegate_hint is False
    assert researcher.search_delegate_hint is False


def test_three_same_topic_triggers_hint(agents):
    agent = _mk_agent()
    for q in (B, B2, B3):
        out = agent._execute_tool("web_search", {"query": q})
        assert out == "mock-result"
    assert agent._search_hint_active is True
    assert agent._search_series == 3
    blocked = agent._execute_tool("web_search", {"query": B2})
    assert "researcher" in blocked


def test_topic_change_resets_series(agents):
    agent = _mk_agent()
    for q in (B, B2):
        agent._execute_tool("web_search", {"query": q})
    assert agent._search_series == 2
    agent._execute_tool("web_search", {"query": T})
    assert agent._search_series == 1
    assert agent._search_hint_active is False


def test_schema_removes_web_search_when_active(agents):
    agent = _mk_agent()
    for q in (B, B2, B3):
        agent._execute_tool("web_search", {"query": q})
    assert agent._search_hint_active is True
    names = [s["function"]["name"] for s in agent._build_tools_schema()]
    assert "web_search" not in names
    assert "final_answer" in names


def test_disabled_by_default(agents):
    model = type("M", (), {"generate": lambda self, m, t=None: ModelResponse(content="ok")})()
    agent = Agent(model=model, tools=[_mk_search_tool()])
    for q in (B, B2, B3):
        agent._execute_tool("web_search", {"query": q})
    assert agent._search_hint_active is False
    assert agent._search_series == 0


def test_reset_after_researcher_delegation(agents):
    main = agents["助手"]
    main._track_search_series({"query": B})
    main._track_search_series({"query": B2})
    main._track_search_series({"query": B3})
    assert main._search_hint_active is True
    with patch.object(main, "_delegate", return_value="总结") as m:
        out = main._execute_tool("researcher", {"task": "查比亚迪股价"})
    assert out == "总结"
    assert m.called
    assert main._search_hint_active is False
    assert main._search_series == 0
    assert main._search_prev_query is None


def test_run_stream_injects_hint_and_forces_delegation(agents):
    class Fake:
        def __init__(self):
            self.calls = []

        def generate(self, messages, tools=None):
            self.calls.append((list(messages), list(tools) if tools else None))
            q = len(self.calls)
            if q <= 3:
                query = [B, B2, B3][q - 1]
                return ModelResponse(
                    tool_calls=[ToolCall(id=f"c{q}", name="web_search", arguments={"query": query})]
                )
            if q == 4:
                return ModelResponse(
                    tool_calls=[ToolCall(id="c4", name="final_answer", arguments={"answer": "完成"})]
                )
            return ModelResponse(content="（无可提取内容）")

    main = agents["助手"]
    main.model = Fake()
    with patch.object(main.tools["web_search"], "func", return_value="mock-result"):
        events = list(main.run_stream("调查比亚迪股价"))
    assert "done" in [e["type"] for e in events]
    calls = main.model.calls
    assert len(calls) >= 4
    fourth_messages = calls[3][0]
    assert any("请调用 researcher 子 Agent" in m.get("content", "") for m in fourth_messages if m["role"] == "system")
    fourth_tools = [t["function"]["name"] for t in calls[3][1]]
    assert "web_search" not in fourth_tools
    assert "researcher" in fourth_tools
