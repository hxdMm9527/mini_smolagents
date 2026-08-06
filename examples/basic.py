import os

from dotenv import load_dotenv

from mini_smolagents import Agent, OpenAIModel, python_interpreter, tool, web_search

load_dotenv()


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    model = OpenAIModel(
        model_id="deepseek-chat",
        temperature=0.0,
    )
    agent = Agent(model=model, tools=[web_search, python_interpreter, get_current_time], max_steps=10)

    result = agent.run("现在是几点？计算 123 * 456 等于多少？")
    print(f"\n最终答案: {result}")


if __name__ == "__main__":
    main()
