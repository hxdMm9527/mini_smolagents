from unittest.mock import MagicMock, patch

from mini_smolagents import Agent, OpenAIModel, python_interpreter, tool, web_search
from mini_smolagents.types import Tool


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
    assert calc.parameters["required"] == ["a"]  # b 有默认值，不在 required
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
    """Mock LLM 返回 final_answer，验证一次就结束"""

    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy])

    # 模拟 LLM 返回: final_answer(answer="result")
    mock_msg = MagicMock()
    mock_msg.content = None

    mock_tc = MagicMock()
    mock_tc.id = "call_test"
    mock_tc.function.name = "final_answer"
    mock_tc.function.arguments = '{"answer": "任务完成"}'

    mock_msg.tool_calls = [mock_tc]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = mock_msg

    model.generate.return_value = mock_response

    result = agent.run("测试任务")
    assert result == "任务完成"
    model.generate.assert_called_once()


def test_agent_run_multi_step():
    """Mock LLM 先调 dummy，再调 final_answer，验证两步"""

    @tool
    def lookup(query: str) -> str:
        return f"查到: {query}"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[lookup])

    # Step 1: lookup
    msg1 = MagicMock()
    msg1.content = None
    tc1 = MagicMock()
    tc1.id = "call_1"
    tc1.function.name = "lookup"
    tc1.function.arguments = '{"query": "test"}'
    msg1.tool_calls = [tc1]

    # Step 2: final_answer
    msg2 = MagicMock()
    msg2.content = None
    tc2 = MagicMock()
    tc2.id = "call_2"
    tc2.function.name = "final_answer"
    tc2.function.arguments = '{"answer": "search result"}'
    msg2.tool_calls = [tc2]

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message = msg1

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message = msg2

    model.generate.side_effect = [resp1, resp2]

    result = agent.run("测试多步任务")
    assert result == "search result"
    assert model.generate.call_count == 2


def test_agent_tool_error_retry():
    """Mock LLM 调一个报错的工具，验证重试"""

    @tool
    def flaky() -> str:
        raise ValueError("总是失败")

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[flaky])

    # Step 1: flaky (失败)
    msg1 = MagicMock()
    msg1.content = None
    tc1 = MagicMock()
    tc1.id = "call_1"
    tc1.function.name = "flaky"
    tc1.function.arguments = "{}"
    msg1.tool_calls = [tc1]

    # Step 2: final_answer (LLM 看到错误后给答案)
    msg2 = MagicMock()
    msg2.content = None
    tc2 = MagicMock()
    tc2.id = "call_2"
    tc2.function.name = "final_answer"
    tc2.function.arguments = '{"answer": "工具失败后仍能继续"}'
    msg2.tool_calls = [tc2]

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message = msg1

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message = msg2

    model.generate.side_effect = [resp1, resp2]

    result = agent.run("测试重试")
    assert result == "工具失败后仍能继续"


def test_agent_max_steps():
    """LLM 永远不调 final_answer，验证超步数报错"""

    @tool
    def loop() -> str:
        return "loop"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[loop], max_steps=3)

    msg = MagicMock()
    msg.content = None
    tc = MagicMock()
    tc.id = "call_x"
    tc.function.name = "loop"
    tc.function.arguments = "{}"
    msg.tool_calls = [tc]

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg

    model.generate.return_value = resp

    try:
        agent.run("死循环任务")
        assert False, "应该抛异常"
    except RuntimeError as e:
        assert "最大步数" in str(e)


def test_trim_messages():
    """验证消息截断：保留 system + 首条 user + 最近 N-2 条"""

    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy], max_messages=5)

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
    assert trimmed[1]["content"] == "task"
    assert trimmed[2]["content"] == "t2"
    assert trimmed[3]["content"] == "a3"
    assert trimmed[4]["content"] == "t3"


def test_trim_messages_not_needed():
    """消息数没超 max_messages，不截断"""

    @tool
    def dummy() -> str:
        return "ok"

    model = MagicMock(spec=OpenAIModel)
    agent = Agent(model=model, tools=[dummy], max_messages=10)

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]

    trimmed = agent._get_trimmed_messages(msgs)
    assert trimmed is msgs


def test_code_agent_extract_code():
    from mini_smolagents.agent import _extract_code

    assert _extract_code("<code>print(1)</code>") == "print(1)"
    assert _extract_code("think...\n<code>\nx = 1\n</code>\n") == "x = 1"
    assert _extract_code("```python\nprint(1)\n```") == "print(1)"
    assert _extract_code("```\nprint(1)\n```") == "print(1)"
    assert _extract_code("no code tags") == "no code tags"


def test_code_agent_final_answer_exception():
    from mini_smolagents.agent import CodeAgent, _StopExec

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

    model = MagicMock(spec=OpenAIModel)
    agent = CodeAgent(model=model, tools=[calc])

    msg = MagicMock()
    msg.content = "<code>\nresult = calc('5 + 3')\nfinal_answer(f'The answer is {result}')\n</code>"
    msg.tool_calls = None

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg

    model.generate.return_value = resp

    result = agent.run("5+3 等于多少？")
    assert "answer is 8" in result


def test_code_agent_code_error():
    """LLM 写了错误代码，Agent 把错误反馈回去，LLM 修正后成功"""

    from mini_smolagents import CodeAgent, tool

    @tool
    def lookup(query: str) -> str:
        return f"result for {query}"

    model = MagicMock(spec=OpenAIModel)
    agent = CodeAgent(model=model, tools=[lookup])

    msg1 = MagicMock()
    msg1.content = "<code>\n1 / 0\n</code>"
    msg1.tool_calls = None

    msg2 = MagicMock()
    msg2.content = "<code>\nresult = lookup('hello')\nfinal_answer(result)\n</code>"
    msg2.tool_calls = None

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message = msg1

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message = msg2

    model.generate.side_effect = [resp1, resp2]

    result = agent.run("test")
    assert "result for hello" in result
