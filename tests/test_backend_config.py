"""backend 记忆接线验证（agents_config.py）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from backend.agents_config import FACTS, build_agents


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
    assert pm.auto_extract_facts is False
    assert developer.auto_extract_facts is False
    assert reviewer.auto_extract_facts is False


def test_backend_subagent_memory(agents):
    main, pm, developer, reviewer = _unpack(agents)
    assert main.memory is not None
    assert pm.memory is not None
    assert developer.memory is pm.memory
    assert reviewer.memory is pm.memory


def test_backend_forget_tool_registered(agents):
    main, pm, developer, reviewer = _unpack(agents)
    assert "forget" in main.tools
    assert "forget" in pm.tools
    assert "forget" in developer.tools