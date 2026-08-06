import os

from openai import OpenAI


class OpenAIModel:
    def __init__(self, model_id: str = "deepseek-chat", api_key: str  | None = None, base_url: str | None = None, **kwargs):
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

    def generate(self, messages: list[dict], tools: list[dict] | None = None):
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=tools,
            **self.client_kwargs,
        )
