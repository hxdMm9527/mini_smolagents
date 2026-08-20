"""backend 记忆接线验证（agents_config.py）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from backend.agents_config import FACTS, RESEARCHER_PROMPT, build_agents


@pytest.fixture(scope="module")
def agents():
    return build_agents()


def _unpack(agents):
    return agents["助手"], agents["PM"], agents["developer"], agents["reviewer"]


def test_backend_facts_shared(agents):
    main, pm, developer, reviewer = _unpack(agents)
    assert main.facts_memory is FACTS
    assert pm.facts_memory is FACTS


def test_backend_extract_only_on_main(agents):
    main, pm, developer, reviewer = _unpack(agents)
    assert main.auto_extract_facts is True
    assert main.experience_memory is not None
    assert main.auto_extract_experience is True
    assert pm.auto_extract_facts is False
    assert developer.auto_extract_facts is False
    assert reviewer.auto_extract_facts is False


def test_backend_subagent_memory(agents):
    main, pm, developer, reviewer = _unpack(agents)
    assert main.memory is not None
    assert pm.memory is not None
    assert developer.memory is pm.memory
    assert reviewer.memory is pm.memory


def _unpack5(agents):
    main, pm, developer, reviewer = _unpack(agents)
    return main, pm, developer, reviewer, agents["researcher"]


def test_backend_researcher_registered(agents):
    main, pm, developer, reviewer, researcher = _unpack5(agents)
    names = [f["function"]["name"] for f in main._build_tools_schema()]
    assert "researcher" in names


def test_backend_researcher_config(agents):
    main, pm, developer, reviewer, researcher = _unpack5(agents)
    tool_names = list(researcher.tools.keys())
    assert "web_search" in tool_names
    assert "python_interpreter" not in tool_names
    assert "多轮精细查询" in RESEARCHER_PROMPT
    assert "researcher" in main.system_prompt
    assert researcher.memory is None


def test_backend_forget_tool_registered(agents):
    main, pm, developer, reviewer = _unpack(agents)
    assert "forget" in main.tools
    assert "forget" in pm.tools
    assert "forget" in developer.tools

def test_researcher_search_disables_semantic_dedup(agents):
    """researcher 的搜索在执行期间关闭语义去重（精细多轮查询不被误拦），结束后恢复。"""
    import mini_smolagents.default_tools as dt
    from mini_smolagents import web_search
    from unittest.mock import patch

    main, pm, developer, reviewer, researcher = _unpack5(agents)
    search_tool = researcher.tools["web_search"]
    assert search_tool is not web_search or search_tool.func.__name__ == "_call"

    with patch("mini_smolagents.default_tools._SEMANTIC_DUP", True) as flag, \
         patch("mini_smolagents.default_tools.web_search.func", return_value="x"):
        pass

    observed = []

    def probe(query):
        observed.append(dt._SEMANTIC_DUP)
        return "probe-result"

    with patch("mini_smolagents.default_tools.web_search.func", side_effect=probe):
        result = search_tool.func("比亚迪 股价 今日 2025")
    assert result == "probe-result"
    assert observed == [False], "执行期间语义去重应被关闭"
    assert dt._SEMANTIC_DUP is True, "执行结束后应恢复为 True"
