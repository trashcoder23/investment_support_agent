"""
news_service.py — Unstructured data service for News and Press Releases.

Responsibilities:
- Fetch recent news for a ticker from NewsData.io (or mock if no API key).
- Fetch or mock press releases.
"""

import os
import requests
from bs4 import BeautifulSoup
import logging
from typing import List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cache_service import cache_manager

logger = logging.getLogger(__name__)

# NewsData.io API Key
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

def fetch_company_news(company_name: str, max_articles: int = 5) -> List[str]:
    """
    Fetch recent news articles about the company using NewsData.io API.
    Returns a list of text chunks representing the news.
    """
    cache_key = f"news_{company_name}_{max_articles}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    if not NEWSDATA_API_KEY:
        logger.warning(f"NEWSDATA_API_KEY not found. Using mock news for {company_name}.")
        mock_news = [
            f"Mock News 1: {company_name} announces strategic partnership to boost AI capabilities.",
            f"Mock News 2: Analysts upgrade {company_name} following strong quarterly guidance.",
            f"Mock News 3: {company_name} launches new product line targeting emerging markets."
        ]
        cache_manager.set(cache_key, mock_news)
        return mock_news

    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={company_name}&language=en"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for item in data.get('results', [])[:max_articles]:
            title = item.get('title', '')
            description = item.get('description', '')
            content = item.get('content', '')
            # Combine available text
            text = f"Title: {title}\nDescription: {description}\nContent: {content}"
            articles.append(text)
            
        if not articles:
            articles = [f"No recent news found for {company_name}."]
            
        cache_manager.set(cache_key, articles)
        return articles
    except Exception as e:
        logger.error(f"Error fetching news for {company_name}: {e}")
        return [f"Error fetching news for {company_name}: {e}"]

def fetch_press_releases(company_name: str) -> List[str]:
    """
    Mock implementation of fetching press releases. 
    In a real app, this would scrape the company's Investor Relations page.
    """
    return [
        f"PRESS RELEASE: {company_name} reports record revenue for Q3 2025.",
        f"PRESS RELEASE: {company_name} declares special dividend of Rs 5 per share."
    ]

def fetch_filings(company_name: str) -> List[str]:
    """Mock implementation for SEC / Company Filings."""
    return [
        f"FILING (10-Q): {company_name} management discussion highlights supply chain improvements.",
        f"FILING (Annual Report): {company_name} outlines sustainability goals for 2030."
    ]

def fetch_earnings_transcripts(company_name: str) -> List[str]:
    """Mock implementation for Earnings Call Transcripts."""
    return [
        f"TRANSCRIPT: CEO of {company_name} mentions 'We expect revenue growth of 12% next quarter.'",
        f"TRANSCRIPT: CFO of {company_name} states 'Operating margins will improve significantly due to cost-cutting initiatives.'"
    ]
