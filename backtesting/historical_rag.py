import os
import logging
from typing import List
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embedding_service import get_embedding_model
from utils.config import VECTORSTORE_DIR

logger = logging.getLogger(__name__)

# Global state to capture telemetry during historical retrieval
RAG_TELEMETRY = {
    "Historical Documents Retrieved": 0,
    "Future Documents Rejected": 0,
    "Documents Passed to LLM": 0,
    "Retrieved Documents": []  # List of metadata dicts for explainability
}

HISTORICAL_DOCUMENTS = [
    # ---- TCS (Tata Consultancy Services) ----
    {
        "company": "TCS",
        "published_date": "2026-01-15",
        "source": "BSE Filing",
        "document_type": "Earnings Release",
        "title": "TCS FY26 Q3 Results",
        "content": "TCS management highlighted a 12% YoY revenue growth. BFSI demand is recovering strongly in North America. 'We expect strong deal wins to continue into Q4,' said the CEO."
    },
    {
        "company": "TCS",
        "published_date": "2026-02-10",
        "source": "TCS Press Center",
        "document_type": "Press Release",
        "title": "TCS signs $1B enterprise contract",
        "content": "Tata Consultancy Services has secured a massive $1 billion contract for digital transformation with a leading European banking group, securing revenue visibility for the next 5 years."
    },
    {
        "company": "TCS",
        "published_date": "2026-03-25",
        "source": "Investor Presentation",
        "document_type": "Guidance",
        "title": "Q4 Margin Outlook Updated",
        "content": "Management indicated that wage hikes and hiring costs might slightly compress operating margins in the upcoming Q4, though revenue guidance remains stable."
    },
    {
        "company": "TCS",
        "published_date": "2026-05-10",
        "source": "Reuters",
        "document_type": "News",
        "title": "TCS Q4 Results Miss Estimates",
        "content": "TCS reported lower-than-expected margins in Q4 due to severe wage inflation. Management reduced FY27 revenue guidance from 10% to 8%."
    },
    
    # ---- HDFC Bank ----
    {
        "company": "HDFC Bank",
        "published_date": "2026-01-18",
        "source": "Earnings Call",
        "document_type": "Earnings Release",
        "title": "HDFC Bank FY26 Q3 earnings",
        "content": "HDFC Bank reported strong retail loan growth of 18% YoY. Deposit growth outpaced credit growth, leading to an improved Credit-Deposit ratio."
    },
    {
        "company": "HDFC Bank",
        "published_date": "2026-02-18",
        "source": "RBI Press Release",
        "document_type": "Regulatory Announcement",
        "title": "RBI policy updates on unsecured lending",
        "content": "The Reserve Bank of India has increased risk weights on unsecured retail loans, forcing banks to increase capital provisioning."
    },
    {
        "company": "HDFC Bank",
        "published_date": "2026-03-12",
        "source": "Management Update",
        "document_type": "Guidance",
        "title": "HDFC Bank Loan Growth Guidance",
        "content": "Due to RBI policy tightening, HDFC Bank management has revised loan growth expectations down to 14% for the next two quarters, prioritizing margin protection."
    },
    {
        "company": "HDFC Bank",
        "published_date": "2026-04-20",
        "source": "Reuters",
        "document_type": "News",
        "title": "HDFC Bank reports NIM compression",
        "content": "Net Interest Margins (NIM) fell by 15 bps in Q4 as the cost of funds rose sharply following the RBI liquidity tightening measures."
    },

    # ---- Infosys ----
    {
        "company": "Infosys",
        "published_date": "2026-01-12",
        "source": "Infosys Newsroom",
        "document_type": "Press Release",
        "title": "Infosys launches specialized AI investments",
        "content": "Infosys announced a $500M investment into expanding its generative AI practice, targeting Fortune 500 manufacturing clients."
    },
    {
        "company": "Infosys",
        "published_date": "2026-02-28",
        "source": "Earnings Call",
        "document_type": "Guidance",
        "title": "Infosys updates Hiring Outlook",
        "content": "Management indicated strong demand, planning to hire 20,000 freshers in FY27 to fulfill recently won large deal pipelines."
    },
    {
        "company": "Infosys",
        "published_date": "2026-03-15",
        "source": "BSE Filing",
        "document_type": "Filing",
        "title": "Major Deal Win with Telecom Giant",
        "content": "Infosys secured a $400M deal to modernize the core billing systems of a leading US telecommunications provider."
    },
    {
        "company": "Infosys",
        "published_date": "2026-05-05",
        "source": "Reuters",
        "document_type": "News",
        "title": "Infosys slashes revenue guidance",
        "content": "Despite earlier deal wins, unexpected project cancellations in the telecom sector forced Infosys to cut its FY27 revenue guidance to just 4-6%."
    },

    # ---- Dr. Reddy's ----
    {
        "company": "Dr. Reddy's",
        "published_date": "2026-01-20",
        "source": "BSE Filing",
        "document_type": "Filing",
        "title": "Dr. Reddy's Q3 Product launches",
        "content": "The company successfully launched 5 new complex generics in the US market, boosting North American revenue by 14%."
    },
    {
        "company": "Dr. Reddy's",
        "published_date": "2026-02-15",
        "source": "FDA Notice",
        "document_type": "Regulatory Announcement",
        "title": "USFDA Form 483 Issued",
        "content": "The USFDA issued a Form 483 with three minor observations for Dr. Reddy's API manufacturing facility in Hyderabad following an inspection."
    },
    {
        "company": "Dr. Reddy's",
        "published_date": "2026-03-30",
        "source": "Management Update",
        "document_type": "Guidance",
        "title": "EBITDA Margin Improvement",
        "content": "Management provided guidance indicating that operating leverage and a better product mix will improve EBITDA margins by 150 bps in the upcoming year."
    },
    {
        "company": "Dr. Reddy's",
        "published_date": "2026-04-25",
        "source": "FDA Notice",
        "document_type": "Regulatory Announcement",
        "title": "USFDA Escalates Warning",
        "content": "The USFDA has escalated the previous Form 483 to an Official Action Indicated (OAI) status, temporarily halting new approvals from the Hyderabad plant."
    }
]

def build_or_load_historical_index() -> FAISS:
    """
    Builds the Historical Knowledge Base once, persists it, and reuses it.
    """
    index_path = os.path.join(VECTORSTORE_DIR, "historical_knowledge_base")
    embeddings = get_embedding_model()
    
    if os.path.exists(os.path.join(index_path, "index.faiss")):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    logger.info("Building Historical Knowledge Base FAISS index...")
    
    docs = []
    for item in HISTORICAL_DOCUMENTS:
        metadata = {
            "company": item["company"],
            "title": item["title"],
            "source": item["source"],
            "document_type": item["document_type"],
            "published_date": item["published_date"]
        }
        text_content = f"[{item['published_date']}] {item['document_type']} - {item['title']} (Source: {item['source']})\nContent: {item['content']}"
        docs.append(Document(page_content=text_content, metadata=metadata))
        
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    
    return vectorstore

def get_historical_relevant_unstructured_data(company_name: str, query: str, cutoff_date: str, top_k: int = 5) -> str:
    """
    Retrieves documents from the historical FAISS index.
    STRICT INTEGRITY CHECK: Filters out any documents where published_date > cutoff_date.
    Logs future documents rejected in RAG_TELEMETRY.
    """
    global RAG_TELEMETRY
    RAG_TELEMETRY = {
        "Historical Documents Retrieved": 0,
        "Future Documents Rejected": 0,
        "Documents Passed to LLM": 0,
        "Retrieved Documents": []
    }
    
    try:
        vectorstore = build_or_load_historical_index()
        
        # Retrieve a large batch to ensure we have enough after filtering
        docs = vectorstore.similarity_search(query, k=top_k * 10)
        
        cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
        
        filtered_docs = []
        for doc in docs:
            # Filter by company
            doc_company = doc.metadata.get("company", "")
            if doc_company.lower() not in company_name.lower() and doc_company.lower() not in query.lower() and company_name != "Unknown":
                continue
                
            RAG_TELEMETRY["Historical Documents Retrieved"] += 1
            
            # STRICT DATE FILTERING
            pub_date_str = doc.metadata.get("published_date")
            if pub_date_str:
                pub_dt = datetime.strptime(pub_date_str, "%Y-%m-%d")
                if pub_dt > cutoff_dt:
                    RAG_TELEMETRY["Future Documents Rejected"] += 1
                    logger.warning(f"Future Document Rejected: {doc.metadata.get('title')} ({pub_date_str}) > cutoff ({cutoff_date})")
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
