"""
embedding_service.py — Wrapping embedding models for RAG.

We use GoogleGenerativeAIEmbeddings since the user uses Gemini 2.x LLM.
"""

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Returns an instance of GoogleGenerativeAIEmbeddings.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set. Embeddings will fail.")
        
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
