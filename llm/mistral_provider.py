import os
from llm.base_provider import BaseLLMProvider
from langchain_openai import ChatOpenAI

class MistralProvider(BaseLLMProvider):
    def get_llm(self) -> ChatOpenAI:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set in environment variables.")
            
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            model="mistral-large-latest",
            temperature=0.2,
            max_tokens=4096
        )
