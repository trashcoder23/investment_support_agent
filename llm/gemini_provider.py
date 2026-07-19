import os
from llm.base_provider import BaseLLMProvider
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiProvider(BaseLLMProvider):
    def get_llm(self) -> ChatGoogleGenerativeAI:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
            
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            api_key=api_key,
            temperature=0.2,
            max_tokens=4096
        )
