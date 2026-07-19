import os
import json
import time
import logging
import re
from utils.config import SECTOR_TICKERS, TICKER_TO_NAME

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class CacheManager:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds

    def _get_cache_path(self, key: str) -> str:
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        return os.path.join(CACHE_DIR, f"{safe_key}.json")

    def get(self, key: str):
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return None
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if time.time() - data.get("timestamp", 0) > self.ttl_seconds:
                logger.info(f"Cache expired for {key}")
                return None
                
            logger.info(f"Cache hit for {key}")
            return data.get("content")
        except Exception as e:
            logger.error(f"Cache read error for {key}: {e}")
            return None

    def set(self, key: str, content):
        path = self._get_cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "content": content}, f)
            logger.info(f"Cached {key}")
        except Exception as e:
            logger.error(f"Cache write error for {key}: {e}")

    def get_age(self, key: str) -> float:
        """Returns age in seconds, or float('inf') if missing."""
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return float('inf')
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return time.time() - data.get("timestamp", 0)
        except:
            return float('inf')

cache_manager = CacheManager(ttl_seconds=1800) # 30 mins

def extract_tickers_from_query(query: str) -> list[str]:
    """
    Extracts tickers based on simple substring/regex matching against SECTOR_TICKERS.
    """
    query_lower = query.lower()
    matched_tickers = set()
    
    # Check for direct ticker mentions
    for sector, companies in SECTOR_TICKERS.items():
        for ticker, name in companies.items():
            # Check ticker symbol (without .NS)
            base_ticker = ticker.split('.')[0].lower()
            if re.search(rf'\b{base_ticker}\b', query_lower):
                matched_tickers.add(ticker)
                
            # Check company name
            name_lower = name.lower()
            if name_lower in query_lower:
                matched_tickers.add(ticker)
                
    return list(matched_tickers)
