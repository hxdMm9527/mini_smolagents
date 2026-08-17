import json

from mini_smolagents.a2a import AgentRegistry, Task
from mini_smolagents.agent import Agent
from mini_smolagents.tools import tool
from mini_smolagents.types import ModelResponse, ToolCall


@tool
def fake_answer(answer) -> str:
    """返回最终答案"""
    return str(answer)


class FakeModel:
    def generate(self, messages, tools=None):
        return ModelResponse(
            content="答",
            tool_calls=[ToolCall(id="1", name="final_answer", arguments={"answer": "42"})],
        )

class ScriptModel:
    """按调用次数依次返回预设响应。"""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def generate(self, messages, tools=None):
        self.calls.append(messages)
        step = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        tool_calls = [
            ToolCall(id=tid, name=name, arguments=json.loads(args))
            for tid, name, args in step.get("tools", [])
        ]
        return ModelResponse(content=step.get("content", ""), tool_calls=tool_calls or None)

def make_agent(name, desc, model=None):
    return Agent(model=model or FakeModel(), tools=[], name=name, description=desc)


def test_register_and_find():
    reg = AgentRegistry()
    dev = make_agent("developer", "写代码", FakeModel())
    reg.register(dev, capabilities=["code", "debug"])

    card = reg.find("developer")
    assert card is not None
    assert card.name == "developer"
    assert card.capabilities == ["code", "debug"]
    assert "final_answer" in card.tools

    assert reg.find("nobody") is None


def test_list_capabilities():
    reg = AgentRegistry()
    reg.register(make_agent("dev", "写代码"), capabilities=["code"])
    reg.register(make_agent("rev", "审代码"), capabilities=["review"])

    caps = reg.list_capabilities()
    assert caps == {"dev": ["code"], "rev": ["review"]}


def test_delegate_success():
    reg = AgentRegistry()
    reg.register(make_agent("dev", "写代码", FakeModel()))
    artifact = reg.delegate(Task(description="写一个函数", target_agent="dev"))
    assert artifact.status == "success"
    assert artifact.content == "42"


def test_delegate_unknown_agent():
    reg = AgentRegistry()
    artifact = reg.delegate(Task(description="x", target_agent="ghost"))
    assert artifact.status == "fail"
    assert "not registered" in artifact.error


def test_delegate_failure_propagates():
    reg = AgentRegistry()

    class Boom:
        name = "boom"
        description = "boom agent"

        def run(self, task):
            raise RuntimeError("boom fail")

    reg.register(Boom())
    artifact = reg.delegate(Task(description="x", target_agent="boom"))
    assert artifact.status == "fail"
    assert "boom fail" in artifact.error


def test_managed_agents_register_and_route():
    reg = AgentRegistry()
    dev = make_agent("developer", "写代码", FakeModel())
    pm = Agent(model=FakeModel(), tools=[], name="PM", description="项目经理",
               managed_agents=[dev], registry=reg)

    assert reg.find("developer") is not None
    assert "developer" in pm.tools
    result = pm.tools["developer"].func(task="写函数")
    assert result == "42"


def test_managed_agents_auto_registry():
    dev = make_agent("developer", "写代码", FakeModel())
    pm = Agent(model=FakeModel(), tools=[], name="PM", description="项目经理",
               managed_agents=[dev])
    assert pm.registry is not None
    assert pm.registry.find("developer") is not None


def test_create_sub_agent_uses_registered():
    reg = AgentRegistry()
    dev = make_agent("developer", "写代码", FakeModel())
    reg.register(dev)

    script = [
        {"content": "拆解", "tools": [("1", "create_sub_agent", '{"name": "developer", "task": "写函数"}')]},
        {"content": "子回答", "tools": [("1", "final_answer", '{"answer": "42"}')]},
        {"content": "主完成", "tools": [("1", "final_answer", '{"answer": "42"}')]},
    ]
    pm = Agent(model=ScriptModel(script), tools=[], name="PM", description="项目经理", registry=reg)
    events = list(pm.run_stream("写一个函数"))
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0]["content"] == "42"
    assert len(reg.list_cards()) == 1


def test_create_sub_agent_creates_and_registers_new():
    script = [
        {"content": "拆解", "tools": [("1", "create_sub_agent", '{"name": "temp", "task": "写函数"}')]},
        {"content": "子回答", "tools": [("1", "final_answer", '{"answer": "42"}')]},
        {"content": "主完成", "tools": [("1", "final_answer", '{"answer": "42"}')]},
    ]
    pm = Agent(model=ScriptModel(script), tools=[], name="PM", description="项目经理")
    events = list(pm.run_stream("写一个函数"))
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0]["content"] == "42"
    assert pm.registry.find("temp") is not None


def test_dynamic_registered_agent_in_schema():
    reg = AgentRegistry()
    pm = Agent(model=FakeModel(), tools=[], name="PM", description="项目经理", registry=reg)
    reg.register(make_agent("scientist", "做数据分析", FakeModel()), capabilities=["analysis"])

    schema = pm._build_tools_schema()
    names = [s["function"]["name"] for s in schema]
    assert "scientist" in names
    entry = next(s for s in schema if s["function"]["name"] == "scientist")
    assert "做数据分析" in entry["function"]["description"]
    assert "analysis" in entry["function"]["description"]
    assert "task" in entry["function"]["parameters"]["properties"]


def test_dynamic_registered_agent_callable():
    reg = AgentRegistry()
    reg.register(make_agent("scientist", "做数据分析", FakeModel()))
    script = [
        {"content": "交给科学家", "tools": [("1", "scientist", '{"task": "分析数据"}')]},
        {"content": "主完成", "tools": [("1", "final_answer", '{"answer": "42"}')]},
    ]
    pm = Agent(model=ScriptModel(script), tools=[], name="PM", description="项目经理", registry=reg)
    events = list(pm.run_stream("分析数据"))
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0]["content"] == "42"


def test_unknown_tool_returns_error():
    reg = AgentRegistry()
    pm = Agent(model=FakeModel(), tools=[], name="PM", description="项目经理", registry=reg)
    result = pm._execute_tool("nobody", {"task": "x"})
    assert result == "Error: unknown tool 'nobody'"


if __name__ == "__main__":
    test_register_and_find()
    test_list_capabilities()
    test_delegate_success()
    test_delegate_unknown_agent()
    test_delegate_failure_propagates()
    test_managed_agents_register_and_route()
    test_managed_agents_auto_registry()
    test_create_sub_agent_uses_registered()
    test_create_sub_agent_creates_and_registers_new()
    test_dynamic_registered_agent_in_schema()
    test_dynamic_registered_agent_callable()
    test_unknown_tool_returns_error()
    print("\n=== ALL A2A TESTS PASSED ===")
