from unittest.mock import MagicMock

from mini_smolagents import Agent, OpenAIModel, python_interpreter, tool, web_search
from mini_smolagents.config import DEFAULT_MAX_FACTS
from mini_smolagents.profile import Profile
from mini_smolagents.types import ModelResponse, Tool, ToolCall


class FakeModel:
    """测试替身：按顺序返回预设的 ModelResponse，无流式接口。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, tools=None):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeModel: 没有更多预设响应")
        return self.responses.pop(0)


def test_tool_decorator():
    @tool
    def calc(a: int, b: float = 1.0) -> str:
        """计算 a 加上 b 的结果"""
        return str(a + b)

    assert isinstance(calc, Tool)
    assert calc.name == "calc"
    assert "计算" in calc.description
    assert calc.parameters["type"] == "object"
    assert "a" in calc.parameters["properties"]
    assert "b" in calc.parameters["properties"]
    assert calc.parameters["properties"]["a"]["type"] == "integer"
    assert calc.parameters["properties"]["b"]["type"] == "number"
    assert calc.parameters["required"] == ["a"]
    assert calc.func(a=3, b=4.0) == "7.0"


def test_builtin_tools_exist():
    assert web_search.name == "web_search"
    assert python_interpreter.name == "python_interpreter"


def test_python_interpreter_eval():
    result = python_interpreter.func(code="5 + 3")
    assert "8" in result


def test_python_interpreter_exec():
    result = python_interpreter.func(code="x = 5\nprint(x + 3)")
    assert "8" in result


def test_python_interpreter_sandbox():
    result = python_interpreter.func(code="__import__('os').system('echo hi')")
    assert "Error" in result or "not allowed" in result.lower()


def test_python_interpreter_timeout():
    result = python_interpreter.func(code="while True: pass")
    assert "timeout" in result.lower() or "timed out" in result.lower()


def test_agent_init_auto_adds_final_answer():
    @tool
    def dummy() -> str:
        """A dummy tool"""
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy])
    assert "final_answer" in agent.tools
    assert "dummy" in agent.tools


def test_agent_run_single_step():
    """FakeModel 返回 final_answer，验证一次就结束"""

    @tool
    def dummy() -> str:
        return "ok"

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="call_test", name="final_answer", arguments={"answer": "任务完成"})]),
    ])
    agent = Agent(model=model, tools=[dummy])

    result = agent.run("测试任务")
    assert result == "任务完成"
    assert len(model.calls) == 1


def test_agent_run_multi_step():
    """FakeModel 先调 lookup，再调 final_answer，验证两步"""

    @tool
    def lookup(query: str) -> str:
        return f"查到: {query}"

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"query": "test"})]),
        ModelResponse(tool_calls=[ToolCall(id="call_2", name="final_answer", arguments={"answer": "search result"})]),
    ])
    agent = Agent(model=model, tools=[lookup])

    result = agent.run("测试多步任务")
    assert result == "search result"
    assert len(model.calls) == 2


def test_agent_tool_error_retry():
    """FakeModel 调一个报错的工具，验证重试"""

    @tool
    def flaky() -> str:
        raise ValueError("总是失败")

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="call_1", name="flaky", arguments={})]),
        ModelResponse(tool_calls=[ToolCall(id="call_2", name="final_answer", arguments={"answer": "工具失败后仍能继续"})]),
    ])
    agent = Agent(model=model, tools=[flaky])

    result = agent.run("测试重试")
    assert result == "工具失败后仍能继续"


def test_agent_max_steps():
    """LLM 永远不调 final_answer，验证超步数报错"""

    @tool
    def loop() -> str:
        return "loop"

    model = FakeModel([ModelResponse(tool_calls=[ToolCall(id="call_x", name="loop", arguments={})])] * 3)
    agent = Agent(model=model, tools=[loop], max_steps=3)

    agent._summarize_messages = MagicMock(return_value="总结：任务未完成")
    result = agent.run("死循环任务")
    agent._summarize_messages.assert_called_once()
    assert result == "总结：任务未完成"


def test_sliding_window():
    """验证滑动窗口：system + 最近 (window_size - 1) 条原文"""

    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy], window_size=5)

    msgs = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "a3"},
        {"role": "tool", "content": "t3"},
    ]

    trimmed = agent._get_trimmed_messages(msgs)

    assert len(trimmed) == 5
    assert trimmed[0]["content"] == "system"
    assert trimmed[1]["content"] == "a2"
    assert trimmed[2]["content"] == "t2"
    assert trimmed[3]["content"] == "a3"
    assert trimmed[4]["content"] == "t3"


def test_sliding_window_not_needed():
    """消息数没超 window_size，不截断"""

    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy], window_size=10)

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]

    trimmed = agent._get_trimmed_messages(msgs)
    assert trimmed is msgs


def test_code_agent_extract_code():
    from mini_smolagents.code_agent import _extract_code

    assert _extract_code("<code>print(1)</code>") == "print(1)"
    assert _extract_code("think...\n<code>\nx = 1\n</code>\n") == "x = 1"
    assert _extract_code("```python\nprint(1)\n```") == "print(1)"
    assert _extract_code("```\nprint(1)\n```") == "print(1)"
    assert _extract_code("no code tags") == "no code tags"


def test_code_agent_final_answer_exception():
    from mini_smolagents.code_agent import CodeAgent, _StopExec

    model = MagicMock(spec=OpenAIModel)
    agent = CodeAgent(model=model, tools=[])

    sandbox, fa = agent._build_sandbox()
    try:
        sandbox["final_answer"]("result_value")
        assert False, "应该抛 _StopExec"
    except _StopExec:
        pass
    assert fa["value"] == "result_value"


def test_code_agent_run_with_final_answer():
    """LLM 生成 final_answer() 代码，Agent 捕获后返回"""

    from mini_smolagents import CodeAgent, tool

    @tool
    def calc(expression: str) -> str:
        return str(eval(expression))

    code = """<code>
result = calc('5 + 3')
final_answer(f'The answer is {result}')
</code>"""
    model = FakeModel([ModelResponse(content=code)])
    agent = CodeAgent(model=model, tools=[calc])

    result = agent.run("5+3 等于多少？")
    assert "answer is 8" in result


def test_code_agent_code_error():
    """LLM 写了错误代码，Agent 把错误反馈回去，LLM 修正后成功"""

    from mini_smolagents import CodeAgent, tool

    @tool
    def lookup(query: str) -> str:
        return f"result for {query}"

    bad_code = """<code>
1 / 0
</code>"""
    good_code = """<code>
result = lookup('hello')
final_answer(result)
</code>"""
    model = FakeModel([
        ModelResponse(content=bad_code),
        ModelResponse(content=good_code),
    ])
    agent = CodeAgent(model=model, tools=[lookup])

    result = agent.run("test")
    assert "result for hello" in result


def test_managed_agent_identity():
    """验证被管理的 Agent 有 name 和 description"""
    model = MagicMock(spec=OpenAIModel)
    sub = Agent(model=model, tools=[], name="helper", description="A helper agent")
    assert sub.name == "helper"
    assert sub.description == "A helper agent"


def test_managed_agent_registration():
    """验证子 Agent 被注册为主 Agent 的 Tool"""

    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    sub = Agent(model=model, tools=[], name="researcher", description="研究助手")

    manager = Agent(model=model, tools=[dummy], managed_agents=[sub])
    assert "researcher" in manager.tools
    assert manager.tools["researcher"].name == "researcher"
    assert manager.tools["researcher"].parameters["required"] == ["task"]


def test_managed_agent_called_by_manager():
    """主 Agent 通过委托调子 Agent，子 Agent 返回结果后主 Agent 综合最终答案"""

    @tool
    def search(q: str) -> str:
        return f"搜索: {q}"

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="call_helper", name="helper", arguments={"task": "找资料"})]),
        ModelResponse(tool_calls=[ToolCall(id="call_sub", name="final_answer", arguments={"answer": "找到的资料"})]),
        ModelResponse(tool_calls=[ToolCall(id="call_done", name="final_answer", arguments={"answer": "任务完成，基于：找到的资料"})]),
    ])

    sub = Agent(model=model, tools=[search], name="helper", description="助手")
    manager = Agent(model=model, tools=[], managed_agents=[sub])

    result = manager.run("复杂任务")
    assert "任务完成" in result
    assert len(model.calls) == 3

def test_create_sub_agent_tool_exists():
    """验证 create_sub_agent 工具自动注入"""

    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy])
    assert "create_sub_agent" in agent.tools
    assert agent.tools["create_sub_agent"].name == "create_sub_agent"
    assert "name" in agent.tools["create_sub_agent"].parameters["properties"]
    assert "task" in agent.tools["create_sub_agent"].parameters["properties"]


def test_create_sub_agent_dynamic_delegation():
    """主 Agent 通过 create_sub_agent 动态创建子 Agent 并拿到结果"""

    @tool
    def search(q: str) -> str:
        return f"搜索: {q}"

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="call_create", name="create_sub_agent", arguments={"name": "helper", "task": "找资料"})]),
        ModelResponse(tool_calls=[ToolCall(id="call_sub", name="final_answer", arguments={"answer": "子任务结果"})]),
        ModelResponse(tool_calls=[ToolCall(id="call_done", name="final_answer", arguments={"answer": "完成"})]),
    ])
    manager = Agent(model=model, tools=[search])

    result = manager.run("复杂任务")
    assert "完成" in result
    assert len(model.calls) == 3

def test_rolling_summary_on_overflow():
    """窗口溢出时，滑出消息被增量摘要，窗口内保持原文。"""

    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy], window_size=4)
    agent._summarize_messages = MagicMock(return_value="摘要块")

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "a3"},
        {"role": "tool", "content": "t3"},
    ]

    ctx = agent._build_context(msgs)

    agent._summarize_messages.assert_called_once()
    assert agent.summary_block == "摘要块"

    assert ctx[0]["role"] == "system"
    assert ctx[1]["role"] == "system" and "摘要块" in ctx[1]["content"]
    assert [m["content"] for m in ctx[2:]] == ["t2", "a3", "t3"]

    # 无新溢出时不重复摘要
    agent._build_context(msgs)
    agent._summarize_messages.assert_called_once()


def test_rolling_summary_incremental():
    """新增溢出只摘要新增部分，摘要块追加而非重写。"""

    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy], window_size=4)
    agent._summarize_messages = MagicMock(side_effect=["摘要A", "摘要B"])

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
    ]
    agent._build_context(msgs)
    assert agent.summary_block == "摘要A"

    msgs2 = msgs + [
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "a3"},
    ]
    agent._build_context(msgs2)
    assert agent._summarize_messages.call_count == 2
    assert agent.summary_block == "摘要A\n\n摘要B"


def test_update_profile_tool_exists():
    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy], profile_dir=".memory")
    assert "update_profile" in agent.tools


def test_update_profile_not_registered_without_profile_dir():
    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy])
    assert "update_profile" not in agent.tools


def test_update_profile_partial_update():
    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy], profile_dir=".memory")
    agent._profile = Profile()
    result = agent.tools["update_profile"].func(facts=["喜欢 Python"])
    assert "facts" in result
    assert agent._profile.data["facts"] == ["喜欢 Python"]
    assert agent._profile.data["name"] == ""


def test_update_profile_final_answer_commits(tmp_path):
    @tool
    def dummy() -> str:
        return "ok"

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="update_profile", arguments={"facts": ["在杭州"]})]),
        ModelResponse(tool_calls=[ToolCall(id="c2", name="final_answer", arguments={"answer": "记住了"})]),
    ])
    agent = Agent(model=model, tools=[dummy], profile_dir=str(tmp_path))
    result = agent.run("记住我在杭州")
    assert result == "记住了"
    loaded = Profile.load(str(tmp_path))
    assert loaded.data["facts"] == ["在杭州"]


def test_update_profile_max_steps_not_committed(tmp_path):
    @tool
    def loop() -> str:
        return "loop"

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="update_profile", arguments={"facts": ["在杭州"]})]),
        ModelResponse(tool_calls=[ToolCall(id="c2", name="loop", arguments={})]),
        ModelResponse(tool_calls=[ToolCall(id="c3", name="loop", arguments={})]),
    ])
    agent = Agent(model=model, tools=[loop], max_steps=3, profile_dir=str(tmp_path))
    agent._summarize_messages = MagicMock(return_value="总结")
    agent.run("测试")
    assert not (tmp_path / "user_profile.json").exists()


def test_sub_agent_no_update_profile():
    @tool
    def search(q: str) -> str:
        return f"搜索: {q}"

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="create_sub_agent", arguments={"name": "helper", "task": "找资料"})]),
        ModelResponse(tool_calls=[ToolCall(id="c2", name="final_answer", arguments={"answer": "子任务结果"})]),
        ModelResponse(tool_calls=[ToolCall(id="c3", name="final_answer", arguments={"answer": "完成"})]),
    ])
    manager = Agent(model=model, tools=[search], profile_dir=".memory")
    assert "update_profile" in manager.tools

    sub_agent, task, err = manager._prepare_sub_agent("helper", "找资料", "")
    assert err is None
    assert "update_profile" not in sub_agent.tools


def test_merge_facts_under_limit_noop():
    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy])
    facts = ["f1", "f2"]
    assert agent._merge_facts(facts) == facts


def test_merge_facts_llm_compress():
    @tool
    def dummy() -> str:
        return "ok"

    model = FakeModel([ModelResponse(content="- 合并1\n- 合并2\n")])
    agent = Agent(model=model, tools=[dummy])
    facts = [f"事实{i}" for i in range(25)]
    result = agent._merge_facts(facts)
    assert result == ["合并1", "合并2"]


def test_merge_facts_failure_truncates():
    @tool
    def dummy() -> str:
        return "ok"

    class FailingModel:
        def generate(self, messages, tools=None):
            raise RuntimeError("llm down")

    agent = Agent(model=FailingModel(), tools=[dummy])
    facts = [f"事实{i}" for i in range(25)]
    result = agent._merge_facts(facts)
    assert len(result) == DEFAULT_MAX_FACTS
    assert result == facts[-DEFAULT_MAX_FACTS:]