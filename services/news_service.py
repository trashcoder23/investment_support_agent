import os
import requests
import logging
from typing import List, Dict
import yfinance as yf
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cache_service import cache_manager

logger = logging.getLogger(__name__)

NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

def fetch_company_news(company_name: str, ticker: str = None, max_articles: int = 10) -> List[Dict]:
    """
    Fetch real unstructured news for a company dynamically.
    Returns a list of dictionaries with: title, content, source, published_date.
    Uses NewsData.io and falls back to Yahoo Finance.
    """
    cache_key = f"dynamic_news_{company_name}_{ticker}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    articles = []

    # 1. Try NewsData.io
    if NEWSDATA_API_KEY:
        try:
            url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={company_name}&language=en"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('results', [])[:max_articles]:
                    pub_date = item.get('pubDate', '')
                    try:
                        pub_date_clean = pub_date.split()[0] if pub_date else datetime.now().strftime("%Y-%m-%d")
                    except:
                        pub_date_clean = datetime.now().strftime("%Y-%m-%d")
                        
                    articles.append({
                        "title": item.get('title', 'No Title'),
                        "content": f"{item.get('description', '')} {item.get('content', '')}".strip(),
                        "source": item.get('source_id', 'NewsData'),
                        "published_date": pub_date_clean
                    })
        except Exception as e:
            logger.warning(f"NewsData.io fetch failed: {e}")

    # 2. Try Yahoo Finance Fallback if not enough articles or no key
    if ticker and len(articles) < max_articles:
        try:
            stock = yf.Ticker(ticker)
            yf_news = stock.news
            if yf_news:
                for item in yf_news[:max_articles]:
                    ts = item.get("providerPublishTime")
                    if ts:
                        pub_date_clean = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    else:
                        pub_date_clean = datetime.now().strftime("%Y-%m-%d")
                        
                    articles.append({
                        "title": item.get("title", "No Title"),
                        "content": item.get("summary", ""),
                        "source": item.get("publisher", "Yahoo Finance"),
                        "published_date": pub_date_clean
                    })
        except Exception as e:
            logger.warning(f"Yahoo Finance news fetch failed: {e}")

    # Clean and remove duplicates
    unique_articles = []
    seen_titles = set()
    for art in articles:
        if art['title'] not in seen_titles and len(art['content']) > 10:
            seen_titles.add(art['title'])
            unique_articles.append(art)
            
    cache_manager.set(cache_key, unique_articles)
    return unique_articles
