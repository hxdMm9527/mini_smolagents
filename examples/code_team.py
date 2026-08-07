from dotenv import load_dotenv
load_dotenv()

from mini_smolagents import Agent, OpenAIModel, python_interpreter, web_search

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
2. 用 create_sub_agent 创建"developer"完成每个子任务，任务描述要详细
3. 用 create_sub_agent 创建"reviewer"审查开发者的代码
4. 如果审核员提出了修改意见，重新委派 developer 修复
5. 最终整合所有审核通过的代码 → 用 final_answer 返回完整项目 + 使用说明
不要自己写代码，分给团队成员做。审核员是必须环节，不能跳过。\
"""

model = OpenAIModel()

developer = Agent(
    model=model,
    tools=[web_search, python_interpreter],
    name="developer",
    description="Python 开发者，写代码和调试。用 python_interpreter 逐步实现功能和测试。",
    system_prompt=DEV_PROMPT,
)

reviewer = Agent(
    model=model,
    tools=[web_search],
    name="reviewer",
    description="代码审核员，审查代码的逻辑错误、安全问题、代码风格，不重写代码。",
    system_prompt=REVIEWER_PROMPT,
)

pm = Agent(
    model=model,
    tools=[web_search, python_interpreter],
    name="PM",
    system_prompt=PM_PROMPT,
    managed_agents=[developer, reviewer],
)

if __name__ == "__main__":
    task = "写一个 Python 函数 validate_email(email: str) -> bool，要求：检查邮箱格式是否合法，用正则表达式实现"
    result = pm.run(task)
    print(f"\n{'='*60}\n最终交付:\n{result}")
