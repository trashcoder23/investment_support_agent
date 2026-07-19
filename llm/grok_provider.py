import os
from llm.base_provider import BaseLLMProvider
from langchain_openai import ChatOpenAI

class GrokProvider(BaseLLMProvider):
    def get_llm(self) -> ChatOpenAI:
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY is not set in environment variables.")
            
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            model="grok-beta",
            temperature=0.2,
            max_tokens=4096
        )
