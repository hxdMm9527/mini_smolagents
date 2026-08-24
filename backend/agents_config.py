"""构建共享的 AgentRegistry 和预配置角色。"""

from dotenv import load_dotenv

load_dotenv()

from mini_smolagents import Agent, AgentRegistry, Checkpoint, EpisodicMemory, ExperienceMemory, FactsMemory, OpenAIModel, Tool, get_current_time, python_interpreter, web_search
from mini_smolagents.config import SUB_AGENT_MAX_STEPS
import mini_smolagents.default_tools as _dt


def _make_researcher_search() -> Tool:
    """researcher 专用搜索：执行期间关闭语义去重（精细多轮查询不被误拦），保留精确匹配。"""
    search_tool = web_search

    def _call(query: str) -> str:
        prev = _dt._SEMANTIC_DUP
        _dt._SEMANTIC_DUP = False
        try:
            return search_tool.func(query)
        finally:
            _dt._SEMANTIC_DUP = prev

    return Tool(
        name=search_tool.name,
        description=search_tool.description,
        parameters=search_tool.parameters,
        func=_call,
    )

REGISTRY = AgentRegistry()
MEMORY = EpisodicMemory(collection_name="agent_memory", persist_dir="./chroma_db")
FACTS = FactsMemory(collection_name="user_facts", persist_dir="./chroma_db")
EXPERIENCE = ExperienceMemory(collection_name="experience_memory", persist_dir="./chroma_db")
CHECKPOINT = Checkpoint(base_dir=".memory")

MAIN_AGENT_NAME = "助手"

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

RESEARCHER_PROMPT = """你是研究员（researcher）。你的任务是通过多轮网页搜索完成查询，并返回结构化总结。
工作流程：
1. 分析查询目标，设计完整精确的搜索词（主题+范围+限定词），不要一次问太杂
2. 搜索后如果信息不足、或需要补充细节（具体数字/日期/来源），可以换更精确的 query 继续搜索——多轮精细查询是你的职责，放心进行
3. 如果搜索结果以 [提示] 开头（命中缓存），说明该主题刚查过、结果一致，不要再换词纠缠，直接用已有结果推进
4. 信息收集完成后用 final_answer 返回结构化总结：要点列表 + 关键数据 + 来源链接，不要遗漏
不要中途停手，直到能回答查询为止。"""


MAIN_PROMPT = """你是用户的专属助手。你的工作方式：
1. 先理解用户的任务，自己决定完成方式，不依赖固定流程
2. 每一轮判断你是否能直接回答：能回答就用 final_answer 返回完整答案，不要直接输出答案文本；不能回答就调用工具获取信息后再判断
3. 需要信息时：优先调用 researcher 研究助手完成搜索（多轮精细查询、返回总结、不占上下文）。web_search 仅供单次简单查询，任何可能需要 2 次以上搜索的主题一律交给 researcher
4. 需要计算/数据处理/写代码：用 python_interpreter 执行
5. 任务需要多次处理/查询，或任务复杂时：调用 create_sub_agent 创建子 Agent 进行，也可以自己一步步完成
6. 判断自己已经能回答时，立即调用 final_answer 结束，不要再生成多余内容
不要为了调用工具而调用工具。能直接回答就直接用 final_answer 回答。"""



def _model(model_id: str = "deepseek-chat"):
    return OpenAIModel(model_id=model_id)


def build_agents() -> dict[str, Agent]:
    model = _model()
    reasoner = _model("deepseek-reasoner")

    developer = Agent(
        model=model,
        tools=[get_current_time, web_search, python_interpreter],
        name="developer",
        description="Python 开发者，写代码和调试。用 python_interpreter 逐步实现功能和测试。",
        system_prompt=DEV_PROMPT,
        registry=REGISTRY,
        memory=MEMORY,
        checkpoint=CHECKPOINT,
    )

    reviewer = Agent(
        model=model,
        tools=[get_current_time, web_search],
        name="reviewer",
        description="代码审核员，审查代码的逻辑错误、安全问题、代码风格，不重写代码。",
        system_prompt=REVIEWER_PROMPT,
        registry=REGISTRY,
        memory=MEMORY,
        checkpoint=CHECKPOINT,
    )

    pm = Agent(
        model=model,
        tools=[get_current_time, web_search, python_interpreter],
        name="PM",
        description="技术项目经理，负责拆解用户需求、调度 developer/reviewer 团队成员完成代码任务并最终整合交付。",
        system_prompt=PM_PROMPT,
        managed_agents=[developer, reviewer],
        registry=REGISTRY,
        memory=MEMORY,
        facts_memory=FACTS,
        checkpoint=CHECKPOINT,
    )

    researcher = Agent(
        model=reasoner,
        tools=[_make_researcher_search(), get_current_time],
        name="researcher",
        description="搜索主入口。可进行多轮精细网页查询（不受缓存拦截），返回精炼总结、不占用你的上下文。需要查资料、查数据、多主题对比、数字精确性要求高的任务，请把搜索交给我。",
        system_prompt=RESEARCHER_PROMPT,
        max_steps=SUB_AGENT_MAX_STEPS,
    )

    main = Agent(
        model=reasoner,
        tools=[get_current_time, web_search, python_interpreter],
        name="助手",
        description="你的专属助手，自己决定如何完成任何任务：直接回答、搜索资料、执行代码或派子助手。",
        system_prompt=MAIN_PROMPT,
        registry=REGISTRY,
        memory=MEMORY,
        facts_memory=FACTS,
        auto_extract_facts=True,
        experience_memory=EXPERIENCE,
        auto_extract_experience=True,
        checkpoint=CHECKPOINT,
        search_delegate_hint=True,
    )

    REGISTRY.register(main, capabilities=["general_assistant", "web_search", "code", "task_decomposition"])
    REGISTRY.register(pm, capabilities=["project_management", "task_decomposition", "code_review_team"])
    REGISTRY.register(developer, capabilities=["code", "debug"])
    REGISTRY.register(reviewer, capabilities=["review"])
    REGISTRY.register(researcher, capabilities=["web_search", "research"])

    return {"助手": main, "PM": pm, "developer": developer, "reviewer": reviewer, "researcher": researcher}


if __name__ == "__main__":
    for card in REGISTRY.list_cards():
        print(card.name, "-", card.description[:60])
