import os
import logging
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embedding_service import get_embedding_model
from utils.config import VECTORSTORE_DIR
from services.news_service import fetch_company_news
from utils.cache_service import cache_manager

logger = logging.getLogger(__name__)

def build_or_load_index(company_name: str, ticker: str = None) -> FAISS:
    """
    Builds a FAISS index for the company using purely dynamic API data.
    """
    index_path = os.path.join(VECTORSTORE_DIR, f"live_{company_name}")
    embeddings = get_embedding_model()
    
    # Check if index exists and is fresh
    cache_key = f"faiss_live_{company_name}_{ticker}"
    index_exists = os.path.exists(os.path.join(index_path, "index.faiss"))
    is_fresh = cache_manager.get_age(cache_key) < cache_manager.ttl_seconds
    
    if index_exists and is_fresh:
        logger.info(f"Loading fresh FAISS index for {company_name}")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    logger.info(f"Building Dynamic FAISS index for {company_name}")
    # Fetch real live data
    articles = fetch_company_news(company_name, ticker)
    
    all_texts = []
    for art in articles:
        all_texts.append(f"Source: {art['source']} ({art['published_date']})\nCompany: {company_name}\nTitle: {art['title']}\nContent: {art['content']}")
            
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    if not all_texts:
        chunks = [f"No unstructured data found for {company_name}."]
    else:
        chunks = text_splitter.split_text("\n\n---\n\n".join(all_texts))
        
    try:
        embeddings.embed_query("test")
        vectorstore = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}")
        
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    cache_manager.set(cache_key, {"status": "built"})
    
    return vectorstore

def get_relevant_unstructured_data(company_name: str, query: str, top_k: int = 5, ticker: str = None) -> str:
    """
    Retrieves the top-k most relevant unstructured data chunks for a query.
    """
    try:
        vectorstore = build_or_load_index(company_name, ticker)
        docs = vectorstore.similarity_search(query, k=top_k)
        if not docs:
            return f"Recent news unavailable for {company_name}."
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Failed to retrieve unstructured data for {company_name}: {e}")
        return f"Recent news unavailable. Recommendation is based on structured financial metrics only."
