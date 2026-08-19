"""记忆核心加固测试：去重计数、删除更新、时效衰减、forget 工具。"""
from datetime import datetime, timedelta

import pytest

from mini_smolagents import Agent
from mini_smolagents.facts import FactsMemory
from mini_smolagents.memory import EpisodicMemory, MemoryHit
from mini_smolagents.types import ModelResponse


class _FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeModel: 没有更多预设响应")
        return self.responses.pop(0)


def _mem(tmp_path):
    return EpisodicMemory(persist_dir=str(tmp_path / "chroma"))


def _facts(tmp_path):
    return FactsMemory(persist_dir=str(tmp_path / "chroma"))


def test_episodic_dedup_counts(tmp_path):
    mem = _mem(tmp_path)
    t1 = f"写个爬虫任务 {datetime.now().isoformat()}"
    r1 = "这是爬虫任务的完整结果内容，长度足够"
    assert mem.add(t1, r1) is True
    assert mem.add(t1, r1) is False
    assert mem.count() == 1
    hits = mem.search(t1, top_k=1)
    assert len(hits) == 1
    assert hits[0].count == 2


def test_episodic_no_false_dedup(tmp_path):
    mem = _mem(tmp_path)
    mem.add("去重字符串函数任务", "结果：使用 dict.fromkeys 实现字符串去重")
    mem.add("部署 FastAPI 服务任务", "结果：安装依赖并配置端口后启动服务")
    assert mem.count() == 2


def test_episodic_delete(tmp_path):
    mem = _mem(tmp_path)
    assert mem.add("写个爬虫任务", "爬虫完整结果内容，长度足够") is True
    assert mem.delete("写个爬虫任务") == 1
    assert mem.count() == 0
    assert mem.delete("写个爬虫任务") == 0


def test_episodic_update_resets(tmp_path):
    mem = _mem(tmp_path)
    assert mem.add("写个爬虫任务", "旧版本完整结果内容，长度足够") is True
    assert mem.update("写个爬虫任务", "Task: 写个爬虫任务\n\nResult: 新版完整结果内容，长度足够") is True
    hits = mem.search("写个爬虫任务", top_k=1)
    assert "新版完整结果" in hits[0].document
    assert hits[0].count == 1
    assert mem.update("不存在的任务 XYZ", "某内容") is False


def test_episodic_decay_orders_recent_first(tmp_path):
    mem = _mem(tmp_path)
    now = datetime.now().isoformat()
    old = (datetime.now() - timedelta(days=400)).isoformat()
    doc = "Task: 排序问题\n\nResult: 用双指针解决的完整方案内容"
    mem.collection.add(documents=[doc] * 2, ids=["old", "new"],
                       metadatas=[{"task": "排序", "timestamp": old, "count": 1},
                                  {"task": "排序", "timestamp": now, "count": 1}])
    hits = mem.search("排序问题", top_k=2)
    assert len(hits) == 2
    assert hits[0].id == "new"
    assert hits[0].score > hits[1].score


def test_memory_hit_defaults():
    hit = MemoryHit(task="t", document="d", score=0.5)
    assert hit.id is None
    assert hit.count == 1
    assert hit.timestamp == ""


def test_forget_tool_deletes_memory(tmp_path):
    model = _FakeModel([ModelResponse(content="ok")])
    mem = _mem(tmp_path)
    mem.add("写个爬虫任务", "爬虫完整结果内容，长度足够")
    agent = Agent(model=model, tools=[], memory=mem)
    assert "forget" in agent.tools
    result = agent.tools["forget"].func("写个爬虫任务")
    assert "已删除 1 条" in result
    assert mem.count() == 0
    result = agent.tools["forget"].func("写个爬虫任务")
    assert "未找到" in result


def test_forget_tool_deletes_facts(tmp_path):
    model = _FakeModel([ModelResponse(content="ok")])
    facts = _facts(tmp_path)
    facts.add("用户是 Python 开发者")
    agent = Agent(model=model, tools=[], facts_memory=facts)
    result = agent.tools["forget"].func("用户是 Python 开发者")
    assert "已删除 1 条" in result
    assert facts.count() == 0


def test_forget_not_registered_without_memory():
    model = _FakeModel([ModelResponse(content="ok")])
    agent = Agent(model=model, tools=[])
    assert "forget" not in agent.tools