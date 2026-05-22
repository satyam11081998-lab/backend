"""
News fetcher service.
Pulls business/economy headlines from GNews and NewsAPI,
deduplicates, and returns a unified list of headline candidates.

This module does NOT classify or score headlines — that's done by
headline_classifier.py. This just pulls raw data.
"""

import os
import httpx
from datetime import datetime, timedelta, timezone
from typing import TypedDict, List, Optional


class RawHeadline(TypedDict):
    """One raw headline from any news API, normalized to a common shape."""
    title: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    source_url: str
    source_name: str
    published_at: str  # ISO 8601 timestamp
    source_api: str    # "gnews" or "newsapi" — for debugging


# Categories we care about for MBA GDs
# Business + economy is where 90% of GD topics come from
GNEWS_TOPICS = ["business", "world"]  # GNews categories
NEWSAPI_CATEGORIES = ["business"]      # NewsAPI categories

# Keywords boost — headlines mentioning these are more likely GD-worthy
# Used downstream by classifier, included here for reference
PRIORITY_KEYWORDS = [
    "RBI", "policy", "GDP", "inflation", "rupee", "tariff",
    "merger", "acquisition", "IPO", "startup", "fintech",
    "regulation", "ban", "tax", "budget", "sector",
    "FDI", "trade", "economy", "stock", "market"
]


def fetch_from_gnews(api_key: str, max_results: int = 15) -> List[RawHeadline]:
    """
    Fetch recent business headlines from GNews API.
    Docs: https://gnews.io/docs/v4
    
    Returns up to max_results headlines. Empty list on failure (does not raise).
    """
    headlines: List[RawHeadline] = []
    
    for topic in GNEWS_TOPICS:
        try:
            params = {
                "category": topic,
                "lang": "en",
                "country": "in",         # India focus
                "max": max_results // len(GNEWS_TOPICS),
                "apikey": api_key,
            }
            response = httpx.get(
                "https://gnews.io/api/v4/top-headlines",
                params=params,
                timeout=15.0
            )
            
            if response.status_code != 200:
                print(f"GNews error for topic '{topic}': HTTP {response.status_code} — {response.text[:200]}")
                continue
            
            data = response.json()
            articles = data.get("articles", [])
            
            for article in articles:
                headlines.append({
                    "title": article.get("title", "").strip(),
                    "description": (article.get("description") or "").strip() or None,
                    "thumbnail_url": article.get("image") or None,
                    "source_url": article.get("url", ""),
                    "source_name": (article.get("source") or {}).get("name", "Unknown"),
                    "published_at": article.get("publishedAt", datetime.now(timezone.utc).isoformat()),
                    "source_api": "gnews",
                })
                
        except Exception as e:
            print(f"GNews fetch failed for topic '{topic}': {type(e).__name__}: {e}")
            continue
    
    return headlines


def fetch_from_newsapi(api_key: str, max_results: int = 15) -> List[RawHeadline]:
    """
    Fetch recent business headlines from NewsAPI.org.
    Docs: https://newsapi.org/docs
    
    Uses /v2/top-headlines (free plan supports this for India).
    Returns up to max_results headlines. Empty list on failure (does not raise).
    """
    headlines: List[RawHeadline] = []
    
    for category in NEWSAPI_CATEGORIES:
        try:
            params = {
                "category": category,
                "country": "in",  # India focus
                "pageSize": max_results,
                "apiKey": api_key,
            }
            response = httpx.get(
                "https://newsapi.org/v2/top-headlines",
                params=params,
                timeout=15.0
            )
            
            if response.status_code != 200:
                print(f"NewsAPI error for category '{category}': HTTP {response.status_code} — {response.text[:200]}")
                continue
            
            data = response.json()
            articles = data.get("articles", [])
            
            for article in articles:
                # Skip articles with no URL (NewsAPI sometimes returns these)
                if not article.get("url"):
                    continue
                    
                headlines.append({
                    "title": (article.get("title") or "").strip(),
                    "description": (article.get("description") or "").strip() or None,
                    "thumbnail_url": article.get("urlToImage") or None,
                    "source_url": article["url"],
                    "source_name": (article.get("source") or {}).get("name", "Unknown"),
                    "published_at": article.get("publishedAt", datetime.now(timezone.utc).isoformat()),
                    "source_api": "newsapi",
                })
                
        except Exception as e:
            print(f"NewsAPI fetch failed for category '{category}': {type(e).__name__}: {e}")
            continue
    
    return headlines


def deduplicate_headlines(headlines: List[RawHeadline]) -> List[RawHeadline]:
    """
    Remove duplicate headlines.
    Duplicates are identified by source_url (exact match).
    Keeps the FIRST occurrence (GNews is checked first, so its version wins).
    """
    seen_urls = set()
    unique: List[RawHeadline] = []
    
    for h in headlines:
        url = h["source_url"]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        unique.append(h)
    
    return unique


def fetch_all_headlines() -> List[RawHeadline]:
    """
    Main entry point: fetch from both APIs, combine, deduplicate.
    
    Reads API keys from environment variables.
    Returns deduplicated list of up to 30 raw headlines.
    Empty list if both APIs fail.
    """
    gnews_key = os.environ.get("GNEWS_API_KEY", "").strip()
    newsapi_key = os.environ.get("NEWSAPI_KEY", "").strip()
    
    all_headlines: List[RawHeadline] = []
    
    if gnews_key:
        gnews_results = fetch_from_gnews(gnews_key)
        all_headlines.extend(gnews_results)
        print(f"GNews returned {len(gnews_results)} headlines")
    else:
        print("WARNING: GNEWS_API_KEY not set, skipping GNews")
    
    if newsapi_key:
        newsapi_results = fetch_from_newsapi(newsapi_key)
        all_headlines.extend(newsapi_results)
        print(f"NewsAPI returned {len(newsapi_results)} headlines")
    else:
        print("WARNING: NEWSAPI_KEY not set, skipping NewsAPI")
    
    unique = deduplicate_headlines(all_headlines)
    print(f"After dedup: {len(unique)} unique headlines")
    
    return unique