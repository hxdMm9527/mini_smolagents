from dotenv import load_dotenv
load_dotenv()

from mini_smolagents import Agent, OpenAIModel, tool, web_search


@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


model = OpenAIModel()
agent = Agent(model=model, tools=[web_search, get_current_time])
print(agent.run("现在几点？顺便查一下今天深圳天气"))
