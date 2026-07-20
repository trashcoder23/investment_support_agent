"""
embedding_service.py — Wrapping embedding models for RAG.
"""

import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings

logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Returns the appropriate embedding model based on the LLM provider.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.warning("MISTRAL_API_KEY not set. Mistral Embeddings will fail.")
        return MistralAIEmbeddings(
            mistral_api_key=api_key,
            model="mistral-embed"
        )
        
    # Default to Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set. Embeddings will fail.")
        
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
