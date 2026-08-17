import json
import os

from openai import OpenAI

from .types import ModelResponse, ToolCall


class OpenAIModel:
    def __init__(self, model_id: str = "deepseek-chat", api_key: str | None = None, base_url: str | None = None, **kwargs):
        self.model_id = model_id
        self.client_kwargs = kwargs

        api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable, "
                "or pass api_key='...' to OpenAIModel()."
            )

        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _parse_arguments(raw: str) -> dict:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _to_response(self, message) -> ModelResponse:
        tcs = None
        if message.tool_calls:
            tcs = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=self._parse_arguments(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]
        return ModelResponse(content=message.content, tool_calls=tcs)

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> ModelResponse:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=tools,
            **self.client_kwargs,
        )
        return self._to_response(resp.choices[0].message)

    def generate_stream(self, messages: list[dict], tools: list[dict] | None = None):
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=tools,
            stream=True,
            **self.client_kwargs,
        )
        tcs: dict[int, dict] = {}
        order: list[int] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                yield ModelResponse(content=delta.content)
            for tc in delta.tool_calls or []:
                idx = tc.index
                if idx not in tcs:
                    tcs[idx] = {"id": "", "name": "", "arguments": ""}
                    order.append(idx)
                if tc.id:
                    tcs[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tcs[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tcs[idx]["arguments"] += tc.function.arguments
        if order:
            parsed = [
                ToolCall(
                    id=tcs[i]["id"],
                    name=tcs[i]["name"],
                    arguments=self._parse_arguments(tcs[i]["arguments"]),
                )
                for i in order
            ]
            yield ModelResponse(content=None, tool_calls=parsed)