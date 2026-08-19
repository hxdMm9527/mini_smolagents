"""经验自动提炼测试：经验库维护、提炼触发/成本闸门、召回、遗忘覆盖。"""
from unittest.mock import MagicMock

from mini_smolagents import Agent
from mini_smolagents.experience import ExperienceMemory
from mini_smolagents.memory import MemoryHit
from mini_smolagents.tools import tool
from mini_smolagents.types import ModelResponse, ToolCall


class _FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeModel: 没有更多预设响应（意外多调用了一次）")
        return self.responses.pop(0)


def _dummy():
    @tool
    def dummy() -> str:
        return "ok"
    return dummy


def test_experience_memory_crud_dedup(tmp_path):
    exp = ExperienceMemory(persist_dir=str(tmp_path / "chroma"))
    text = "写爬虫时优先用 Playwright 处理动态页面，静态页用 requests 即可"
    assert exp.add(text) is True
    assert exp.add(text) is False
    assert exp.count() == 1
    hits = exp.search("爬虫", top_k=1)
    assert len(hits) == 1
    assert hits[0].count == 2
    assert "Task: 经验" not in hits[0].document
    assert exp.update("爬虫", "新版经验：优先 Playwright 动态页面") is True
    assert "新版经验" in exp.search("爬虫", top_k=1)[0].document
    assert exp.delete("新版经验") == 1
    assert exp.count() == 0
    assert exp.delete("新版经验") == 0


def test_experience_distill_triggered_on_high_count():
    model = _FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="final_answer", arguments={"answer": "完成"})]),
        ModelResponse(content="写爬虫时优先用 Playwright 处理动态页面，静态页用 requests 即可。"),
    ])
    mem = MagicMock()
    mem.search.return_value = [
        MemoryHit(task="写个爬虫任务", document="Task: 写个爬虫任务\n\nResult: 用 requests 抓取成功",
                  score=0.8, count=2),
    ]
    exp = MagicMock()
    exp.add.return_value = True
    agent = Agent(model=model, tools=[_dummy()], memory=mem, experience_memory=exp,
                  auto_extract_experience=True)
    result = agent.run("写个爬虫任务")
    assert result == "完成"
    assert exp.add.call_count == 1
    text = exp.add.call_args[0][0]
    assert "Playwright" in text


def test_experience_no_distill_when_count_low():
    model = _FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="final_answer", arguments={"answer": "完成"})]),
    ])
    mem = MagicMock()
    mem.search.return_value = [
        MemoryHit(task="写个爬虫任务", document="Task: 写个爬虫任务\n\nResult: 抓取成功",
                  score=0.8, count=1),
    ]
    exp = MagicMock()
    agent = Agent(model=model, tools=[_dummy()], memory=mem, experience_memory=exp,
                  auto_extract_experience=True)
    agent.run("写个爬虫任务")
    assert exp.add.call_count == 0
    assert len(model.calls) == 1


def test_experience_recall_in_context():
    mem = MagicMock()
    mem.search.return_value = []
    exp = MagicMock()
    exp.search.return_value = [
        MemoryHit(task="经验", document="写爬虫优先用 Playwright 处理动态页面", score=0.9),
    ]
    agent = Agent(model=_FakeModel([]), tools=[_dummy()], memory=mem, experience_memory=exp)
    text = agent._recall_text("爬虫")
    assert "过去经验" in text
    assert "Playwright" in text


def test_forget_covers_experience():
    exp = MagicMock()
    exp.delete.return_value = 1
    agent = Agent(model=_FakeModel([ModelResponse(content="ok")]), tools=[_dummy()],
                  experience_memory=exp)
    assert "forget" in agent.tools
    result = agent.tools["forget"].func("爬虫")
    assert "已删除 1 条" in result
    exp.delete.assert_called_once_with("爬虫")