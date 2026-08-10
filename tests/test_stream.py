import tempfile
from types import SimpleNamespace
from pathlib import Path

from mini_smolagents.agent import Agent
from mini_smolagents.memory import Checkpoint, EpisodicMemory
from mini_smolagents.tools import tool


@tool
def fake_search(query: str) -> str:
    """搜索测试"""
    return "fake search result"


class FakeModel:
    """按调用次数依次返回预设的 LLM 响应。"""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def generate(self, messages, tools=None):
        self.calls.append(messages)
        step = self.script[min(len(self.calls) - 1, len(self.script) - 1)]

        def _tc(name, args, tid):
            return SimpleNamespace(
                id=tid,
                type="function",
                function=SimpleNamespace(name=name, arguments=args),
            )

        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=step.get("content", ""),
                    tool_calls=[_tc(name, args, tid) for tid, name, args in step.get("tools", [])] or None,
                )
            )]
        )


def collect_stream(agent, task):
    return list(agent.run_stream(task))


def test_event_sequence(tmpdir):
    script = [
        {
            "content": "我先搜索",
            "tools": [("1", "fake_search", '{"query": "test"}')],
        },
        {
            "content": "完成，给答案",
            "tools": [("2", "final_answer", '{"answer": "42"}')],
        },
    ]
    agent = Agent(model=FakeModel(script), tools=[fake_search], name="tester")
    events = collect_stream(agent, "测试任务")

    types = [e["type"] for e in events]
    assert types == ["step", "thought", "action", "result", "step", "thought", "action", "result", "done"], types

    assert events[0]["step"] == 1
    assert events[1]["content"] == "我先搜索"
    assert events[2]["tool"] == "fake_search"
    assert events[2]["args"] == {"query": "test"}
    assert events[3]["content"] == "fake search result"
    assert events[-1]["type"] == "done"
    assert events[-1]["content"] == "42"
    assert events[-1]["stored"] is False


def test_memory_injection_and_store(tmpdir):
    mem = EpisodicMemory(collection_name="t_stream", persist_dir=str(tmpdir / "chroma"))
    mem.add("之前的任务", "之前的结果内容")

    script = [
        {"content": "直接答", "tools": [("1", "final_answer", '{"answer": "这是完整答案内容值得存储"}')]},
    ]
    agent = Agent(model=FakeModel(script), tools=[], name="tester", memory=mem)
    events = collect_stream(agent, "新任务")

    assert events[-1]["stored"] is True
    assert mem.count() == 2
    found = mem.search("新任务", top_k=1)
    assert len(found) == 1
    assert "新任务" in found[0]["document"]


def test_memory_injection_into_system_prompt(tmpdir):
    mem = EpisodicMemory(collection_name="t_inject", persist_dir=str(tmpdir / "chroma"))
    mem.add("之前关于邮箱的讨论", "之前的结果")

    class CaptureModel(FakeModel):
        def generate(self, messages, tools=None):
            self.system_content = messages[0]["content"]
            return super().generate(messages, tools)

    script = [{"content": "答", "tools": [("1", "final_answer", '{"answer": "x"}')]}]
    model = CaptureModel(script)
    agent = Agent(model=model, tools=[], name="tester", memory=mem)
    collect_stream(agent, "邮箱验证")

    assert "相关历史记忆" in model.system_content
    assert "邮箱" in model.system_content


def test_checkpoint_saved(tmpdir):
    cp = Checkpoint(base_dir=str(tmpdir / "cp"))
    script = [{"content": "答", "tools": [("1", "final_answer", '{"answer": "y"}')]}]
    agent = Agent(
        model=FakeModel(script),
        tools=[],
        name="tester",
        checkpoint=cp,
        session_id="session_stream_1",
    )
    collect_stream(agent, "任务")
    loaded = cp.load("session_stream_1")
    assert loaded is not None
    assert loaded[0]["role"] == "system"


def test_run_returns_result(tmpdir):
    script = [{"content": "答", "tools": [("1", "final_answer", '{"answer": "最终答案"}')]}]
    agent = Agent(model=FakeModel(script), tools=[], name="tester")
    result = agent.run("任务")
    assert result == "最终答案"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        base = Path(d)
        test_event_sequence(base)
        test_memory_injection_and_store(base)
        test_memory_injection_into_system_prompt(base)
        test_checkpoint_saved(base)
        test_run_returns_result(base)
        print("\n=== ALL STREAM TESTS PASSED ===")
