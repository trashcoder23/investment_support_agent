import os
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from llm.grok_provider import GrokProvider
from llm.gemini_provider import GeminiProvider
from llm.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

def get_llm_model() -> BaseChatModel:
    """
    Resolves and returns the configured ChatModel.
    Falls back to Gemini if the preferred provider fails or has missing credentials.
    """
    provider = os.getenv("LLM_PROVIDER", "grok").lower()
    
    try:
        if provider == "grok":
            logger.info("Initializing Grok provider...")
            return GrokProvider().get_llm()
        elif provider == "openai":
            logger.info("Initializing OpenAI provider...")
            return OpenAIProvider().get_llm()
        elif provider == "gemini":
            logger.info("Initializing Gemini provider...")
            return GeminiProvider().get_llm()
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM provider '{provider}': {e}. Falling back to Gemini...")
        try:
            return GeminiProvider().get_llm()
        except Exception as fallback_err:
            logger.error(f"Fallback to Gemini failed: {fallback_err}")
            raise fallback_err
