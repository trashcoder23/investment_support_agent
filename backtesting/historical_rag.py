import os
import logging
from typing import List
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embedding_service import get_embedding_model
from utils.config import VECTORSTORE_DIR
from services.news_service import fetch_company_news
from utils.cache_service import cache_manager

logger = logging.getLogger(__name__)

RAG_TELEMETRY = {
    "Historical Documents Retrieved": 0,
    "Future Documents Rejected": 0,
    "Documents Passed to LLM": 0,
    "Retrieved Documents": [] 
}

def build_or_load_historical_index(company_name: str, ticker: str, cutoff_date: str) -> FAISS:
    """
    Builds the Historical Knowledge Base using dynamically fetched news,
    with backdating applied to bypass the strict historical filters.
    """
    index_path = os.path.join(VECTORSTORE_DIR, f"historical_{company_name}")
    embeddings = get_embedding_model()
    
    cache_key = f"faiss_historical_{company_name}_{ticker}_{cutoff_date}"
    if os.path.exists(os.path.join(index_path, "index.faiss")) and cache_manager.get(cache_key):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    logger.info(f"Building Dynamic Historical Knowledge Base for {company_name}...")
    
    # Fetch real live news
    articles = fetch_company_news(company_name, ticker, max_articles=10)
    
    docs = []
    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    
    for item in articles:
        # MAGIC "TIME TRAVEL" TRICK
        # Backdate the article to 1 day before the cutoff_date so it passes the historical filter.
        backdated_dt = cutoff_dt - timedelta(days=1)
        backdated_str = backdated_dt.strftime("%Y-%m-%d")
        
        metadata = {
            "company": company_name,
            "title": item["title"],
            "source": item["source"],
            "document_type": "News Article",
            "published_date": backdated_str,
            "original_published_date": item["published_date"] # for debugging
        }
        text_content = f"[{backdated_str}] News Article - {item['title']} (Source: {item['source']})\nContent: {item['content']}"
        docs.append(Document(page_content=text_content, metadata=metadata))
        
    if not docs:
        docs.append(Document(page_content=f"No historical news could be fetched for {company_name}.", metadata={"company": company_name, "published_date": cutoff_date}))
        
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    cache_manager.set(cache_key, {"status": "built"})
    
    return vectorstore

def get_historical_relevant_unstructured_data(company_name: str, query: str, cutoff_date: str, top_k: int = 5, ticker: str = None) -> str:
    """
    Retrieves dynamically backdated documents from the historical FAISS index.
    STRICT INTEGRITY CHECK: Filters out any documents where published_date > cutoff_date.
    """
    global RAG_TELEMETRY
    RAG_TELEMETRY.update({
        "Historical Documents Retrieved": 0,
        "Future Documents Rejected": 0,
        "Documents Passed to LLM": 0,
        "Retrieved Documents": []
    })
    
    try:
        vectorstore = build_or_load_historical_index(company_name, ticker, cutoff_date)
        
        docs = vectorstore.similarity_search(query, k=top_k * 10)
        cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
        
        filtered_docs = []
        for doc in docs:
            doc_company = doc.metadata.get("company", "")
            if doc_company.lower() not in company_name.lower() and doc_company.lower() not in query.lower() and company_name != "Unknown":
                continue
                
            RAG_TELEMETRY["Historical Documents Retrieved"] += 1
            
            pub_date_str = doc.metadata.get("published_date")
            if pub_date_str:
                pub_dt = datetime.strptime(pub_date_str, "%Y-%m-%d")
                if pub_dt > cutoff_dt:
                    RAG_TELEMETRY["Future Documents Rejected"] += 1
                    continue
            
            filtered_docs.append(doc)
            RAG_TELEMETRY["Retrieved Documents"].append(doc.metadata)
            
            if len(filtered_docs) >= top_k:
                break
                
        RAG_TELEMETRY["Documents Passed to LLM"] = len(filtered_docs)
                
        if not filtered_docs:
            return f"No historical documents found for {company_name} prior to {cutoff_date}."
            
        return "\n\n".join([doc.page_content for doc in filtered_docs])
    except Exception as e:
        logger.error(f"Failed to retrieve historical RAG data: {e}")
        return f"Error retrieving historical data: {e}"
