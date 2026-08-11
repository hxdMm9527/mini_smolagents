"""构建共享的 AgentRegistry 和预配置角色。"""

from dotenv import load_dotenv

load_dotenv()

from mini_smolagents import Agent, AgentRegistry, Checkpoint, EpisodicMemory, OpenAIModel, python_interpreter, web_search

REGISTRY = AgentRegistry()
MEMORY = EpisodicMemory(collection_name="agent_memory", persist_dir="./chroma_db")
CHECKPOINT = Checkpoint(base_dir=".memory")

DEV_PROMPT = """\
你是资深 Python 开发者。你的开发流程：
1. 理解需求，分析需要什么函数和类
2. 用 python_interpreter 逐步实现代码，每写完一段就测试
3. 遇到不确定的 API/库用法，先 web_search 查官方文档
4. 代码要求：清晰的命名、完善的异常处理、必要的类型标注
5. 完成后用 final_answer 返回完整可运行的代码\
"""

REVIEWER_PROMPT = """\
你是代码审核员。只审查不重写。审查清单：
1. 逻辑错误：边界条件、空值、异常处理是否完善
2. 安全问题：输入校验是否充分
3. 代码风格：命名是否清晰、函数是否过长
4. 返回审查意见列表，每条意见标注严重程度（严重/建议）。
5. 如果代码没问题，返回"审核通过"。
6. 用 final_answer 返回结果。\
"""

PM_PROMPT = """\
你是技术项目经理。工作流程：
1. 仔细分析用户需求，拆解为清晰的独立子任务
2. 直接调用 developer 工具完成每个子任务（developer 是预注册的团队成员，给它详细的任务描述）
3. 调用 reviewer 工具审查开发者的代码
4. 如果审核员提出了修改意见，重新调用 developer 修复
5. 最终整合所有审核通过的代码 → 用 final_answer 返回完整项目 + 使用说明
不要自己写代码，分给团队成员做。审核员是必须环节，不能跳过。\
"""

MAIN_PROMPT = """\
你是用户的专属助手。你的工作方式：
1. 先理解用户的任务，自己决定完成方式，不依赖固定流程
2. 简单任务（问候、闲聊、直接可答的问题）：直接给出答案，调用 final_answer 结束
3. 需要信息：用 web_search 搜索
4. 需要计算/数据处理/写代码：用 python_interpreter 执行
5. 任务复杂时：可以拆解并调用 create_sub_agent 创建子助手分担，也可以自己一步步完成
6. 完成任务后用 final_answer 返回最终结果
不要为了调用工具而调用工具。能直接回答就直接回答。\
"""


def _model():
    return OpenAIModel()


def build_agents() -> dict[str, Agent]:
    model = _model()

    developer = Agent(
        model=model,
        tools=[web_search, python_interpreter],
        name="developer",
        description="Python 开发者，写代码和调试。用 python_interpreter 逐步实现功能和测试。",
        system_prompt=DEV_PROMPT,
        registry=REGISTRY,
        checkpoint=CHECKPOINT,
    )

    reviewer = Agent(
        model=model,
        tools=[web_search],
        name="reviewer",
        description="代码审核员，审查代码的逻辑错误、安全问题、代码风格，不重写代码。",
        system_prompt=REVIEWER_PROMPT,
        registry=REGISTRY,
        checkpoint=CHECKPOINT,
    )

    pm = Agent(
        model=model,
        tools=[web_search, python_interpreter],
        name="PM",
        description="技术项目经理，负责拆解用户需求、调度 developer/reviewer 团队成员完成代码任务并最终整合交付。",
        system_prompt=PM_PROMPT,
        managed_agents=[developer, reviewer],
        registry=REGISTRY,
        memory=MEMORY,
        checkpoint=CHECKPOINT,
    )

    main = Agent(
        model=model,
        tools=[web_search, python_interpreter],
        name="助手",
        description="你的专属助手，自己决定如何完成任何任务：直接回答、搜索资料、执行代码或派子助手。",
        system_prompt=MAIN_PROMPT,
        registry=REGISTRY,
        memory=MEMORY,
        checkpoint=CHECKPOINT,
    )

    REGISTRY.register(main, capabilities=["general_assistant", "web_search", "code", "task_decomposition"])
    REGISTRY.register(pm, capabilities=["project_management", "task_decomposition", "code_review_team"])
    REGISTRY.register(developer, capabilities=["code", "debug"])
    REGISTRY.register(reviewer, capabilities=["review"])

    return {"助手": main, "PM": pm, "developer": developer, "reviewer": reviewer}


if __name__ == "__main__":
    for card in REGISTRY.list_cards():
        print(card.name, "-", card.description[:60])
