import os
from llm.base_provider import BaseLLMProvider
from langchain_openai import ChatOpenAI

class OpenAIProvider(BaseLLMProvider):
    def get_llm(self) -> ChatOpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
            
        return ChatOpenAI(
            api_key=api_key,
            model="gpt-4o",
            temperature=0.2,
            max_tokens=4096
        )
