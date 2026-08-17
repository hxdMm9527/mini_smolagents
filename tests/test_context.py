from mini_smolagents.context import ContextComposer, SessionMetadata, estimate_tokens


def test_session_metadata():
    m = SessionMetadata(session_id="s1", agent_name="a", model="deepseek-chat")
    assert m.session_id == "s1"
    assert m.agent_name == "a"
    assert m.model == "deepseek-chat"
    assert m.topic is None
    assert m.started_at


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 2
    assert estimate_tokens("你好") == 1


def test_compose_order():
    composer = ContextComposer(token_budget=10000)
    msgs = composer.compose(
        system_prompt="系统指令",
        profile="用户档案",
        recall="记忆召回",
        summary="历史摘要",
        window=[{"role": "user", "content": "当前问题"}],
    )
    assert msgs[0]["role"] == "system"
    assert "系统指令" in msgs[0]["content"]
    assert "用户档案" in msgs[0]["content"]
    assert "记忆召回" in msgs[0]["content"]
    # 顺序：档案在召回前
    assert msgs[0]["content"].index("用户档案") < msgs[0]["content"].index("记忆召回")
    # 摘要层在窗口前
    assert msgs[1]["role"] == "system" and "历史摘要" in msgs[1]["content"]
    assert msgs[2]["role"] == "user"


def test_compose_skips_empty_layers():
    composer = ContextComposer()
    msgs = composer.compose(system_prompt="S", window=[{"role": "user", "content": "u"}])
    assert len(msgs) == 2
    assert "档案" not in msgs[0]["content"]
    assert "召回" not in msgs[0]["content"]


def test_compose_token_budget_truncates_summary_then_recall():
    composer = ContextComposer(token_budget=30)
    summary = "A" * 100
    recall = "B" * 100
    msgs = composer.compose(system_prompt="S", recall=recall, summary=summary)
    # 摘要被截断，召回被完全截断
    assert "B" not in msgs[0]["content"]
    summary_content = msgs[1]["content"]
    assert len(summary_content) < len(summary)
    assert "…" in summary_content


def test_compose_window_never_truncated():
    composer = ContextComposer(token_budget=10)
    window = [{"role": "user", "content": "原文内容不能丢" * 10}]
    msgs = composer.compose(system_prompt="S", summary="A" * 100, window=window)
    assert msgs[-1] == window[0]


def test_compose_profile_after_system_prompt():
    composer = ContextComposer()
    msgs = composer.compose(
        system_prompt="系统指令",
        profile="档案内容",
        recall="召回内容",
        window=[{"role": "user", "content": "u"}],
    )
    content = msgs[0]["content"]
    assert content.index("系统指令") < content.index("档案内容") < content.index("召回内容")