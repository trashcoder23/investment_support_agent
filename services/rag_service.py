"""
rag_service.py — Retrieval-Augmented Generation logic.

Responsibilities:
- Build or reuse FAISS index for unstructured data.
- Retrieve top-K relevant chunks for a given query.
"""

import os
import logging
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embedding_service import get_embedding_model
from utils.config import VECTORSTORE_DIR
from services.news_service import fetch_company_news, fetch_press_releases, fetch_filings, fetch_earnings_transcripts
from utils.cache_service import cache_manager

logger = logging.getLogger(__name__)

def build_or_load_index(company_name: str) -> FAISS:
    """
    Builds a FAISS index for the company if it doesn't exist, otherwise loads it.
    """
    index_path = os.path.join(VECTORSTORE_DIR, company_name)
    embeddings = get_embedding_model()
    
    # Check if index exists and is fresh
    cache_key = f"faiss_{company_name}"
    index_exists = os.path.exists(os.path.join(index_path, "index.faiss"))
    is_fresh = cache_manager.get_age(cache_key) < cache_manager.ttl_seconds
    
    if index_exists and is_fresh:
        logger.info(f"Loading fresh FAISS index for {company_name}")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    logger.info(f"Building/Rebuilding FAISS index for {company_name}")
    # 1. Fetch all unstructured data
    news = fetch_company_news(company_name)
    press = fetch_press_releases(company_name)
    filings = fetch_filings(company_name)
    transcripts = fetch_earnings_transcripts(company_name)
    
    all_texts = []
    
    for category, items in [("News", news), ("Press Release", press), ("Filing", filings), ("Transcript", transcripts)]:
        for item in items:
            all_texts.append(f"Source: {category}\nCompany: {company_name}\nContent: {item}")
            
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text("\n\n---\n\n".join(all_texts))
    
    # 3. Verify embeddings and create FAISS index
    if not chunks:
        chunks = [f"No unstructured data found for {company_name}."]
        
    try:
        # Verify the embedding model works before generating the full index
        embeddings.embed_query("test")
        vectorstore = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise RuntimeError(f"Embedding generation failed. Check your API key or model availability. Details: {e}")
        
    # 4. Save index and mark cache as fresh
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    cache_manager.set(cache_key, {"status": "built"})
    
    return vectorstore

def get_relevant_unstructured_data(company_name: str, query: str, top_k: int = 5) -> str:
    """
    Retrieves the top-k most relevant unstructured data chunks for a query.
    """
    try:
        vectorstore = build_or_load_index(company_name)
        docs = vectorstore.similarity_search(query, k=top_k)
        if not docs:
            return f"Recent news unavailable for {company_name}."
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Failed to retrieve unstructured data for {company_name}: {e}")
        return f"Recent news unavailable. Recommendation is based on structured financial metrics only. Confidence reduced. (Error: {e})"
