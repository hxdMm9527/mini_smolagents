import json

from .default_tools import final_answer as _FINAL_ANSWER_TOOL

SYSTEM_PROMPT = """\
你是一个善于逐步解决问题的助手。你可以使用工具调用来完成任务。
当你有最终答案时，调用 `final_answer` 工具。
不要在相同的参数下重复调用同一个工具。\
"""


class Agent:
    def __init__(self, model, tools, max_steps=10):
        self.model = model
        self.max_steps = max_steps
        self.tools = {}

        for t in tools:
            self.tools[t.name] = t

        if "final_answer" not in self.tools:
            self.tools["final_answer"] = _FINAL_ANSWER_TOOL

    def _build_tools_schema(self):
        schema = []
        for t in self.tools.values():
            schema.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        return schema

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        tools_schema = self._build_tools_schema()

        for step in range(1, self.max_steps + 1):
            print(f"--- Step {step}/{self.max_steps} ---")

            response = self.model.generate(messages, tools_schema)
            msg = response.choices[0].message

            if not msg.tool_calls:
                text = msg.content or ""
                print(f"  Thought: {text}")
                messages.append({"role": "assistant", "content": text})
                continue

            tool_calls_dict = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": tool_calls_dict,
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                raw_args = tc.function.arguments
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

                result = None
                for attempt in range(1, 4):
                    try:
                        result = self.tools[tool_name].func(**args)
                        break
                    except Exception as e:
                        if attempt == 3:
                            result = f"Error after 3 retries: {type(e).__name__}: {e}"

                print(f"  Action: {tool_name}({args})")
                print(f"  Result: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

                if tool_name == "final_answer":
                    print(f"Done.")
                    return str(result)

        raise RuntimeError(
            f"Agent 达到最大步数（{self.max_steps}）但未完成任务。"
        )
