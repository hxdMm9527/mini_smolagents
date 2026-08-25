from dotenv import load_dotenv
load_dotenv()

from mini_smolagents import Agent, OpenAIModel, get_current_time, web_search, python_interpreter

model = OpenAIModel(model_id="deepseek-reasoner")
agent = Agent(model=model, tools=[get_current_time, web_search, python_interpreter], max_steps=12)
print(agent.run("调查特斯拉和比亚迪今天的股价，计算两者的涨跌幅"))
