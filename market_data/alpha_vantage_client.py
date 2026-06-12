"""
market_data/alpha_vantage_client.py

Alpha Vantage API client with in-process TTL cache.

Endpoints used:
  GLOBAL_QUOTE   — current price, change, change %
  OVERVIEW       — P/E, EPS, market cap, 52-week high/low, forward P/E, dividend yield
  NEWS_SENTIMENT — latest news with per-ticker sentiment scores

Free tier limits: 25 requests/day, 5 requests/minute.
The TTL cache (default 60 s) prevents quota burns from repeated Streamlit renders.
"""

import logging
import time
from typing import Any, Optional

import requests

from config.settings import ALPHA_VANTAGE_API_KEY, MARKET_DATA_CACHE_TTL

_BASE_URL = "https://www.alphavantage.co/query"
_cache: dict[str, tuple[float, Any]] = {}

logger = logging.getLogger("alpha_vantage")


def _cached_get(function: str, symbol: str, **extra_params) -> dict:
    cache_key = f"{function}:{symbol}"
    now = time.time()

    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if now - ts < MARKET_DATA_CACHE_TTL:
            return data

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        **extra_params,
    }
    response = requests.get(_BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    _cache[cache_key] = (now, data)
    return data


def get_quote(ticker: str) -> Optional[dict]:
    """
    Returns {price, previous_close, change, change_percent} or None on error.
    """
    try:
        data = _cached_get("GLOBAL_QUOTE", ticker)
        q = data.get("Global Quote", {})
        if not q:
            return None
        return {
            "price": q.get("05. price", "N/A"),
            "previous_close": q.get("08. previous close", "N/A"),
            "change": q.get("09. change", "N/A"),
            "change_percent": q.get("10. change percent", "N/A"),
        }
    except Exception as exc:
        logger.warning("get_quote failed for %s: %s", ticker, exc)
        return None


def get_fundamentals(ticker: str) -> Optional[dict]:
    """
    Returns {pe_ratio, eps, market_cap, week_52_high, week_52_low,
    forward_pe, dividend_yield} or None on error.
    """
    try:
        data = _cached_get("OVERVIEW", ticker)
        if not data or "Symbol" not in data:
            return None
        return {
            "pe_ratio": data.get("PERatio", "N/A"),
            "eps": data.get("EPS", "N/A"),
            "market_cap": data.get("MarketCapitalization", "N/A"),
            "week_52_high": data.get("52WeekHigh", "N/A"),
            "week_52_low": data.get("52WeekLow", "N/A"),
            "forward_pe": data.get("ForwardPE", "N/A"),
            "dividend_yield": data.get("DividendYield", "N/A"),
        }
    except Exception as exc:
        logger.warning("get_fundamentals failed for %s: %s", ticker, exc)
        return None


def get_news_sentiment(ticker: str) -> list[dict]:
    """
    Returns up to 10 news items, each with title, published, sentiment_score,
    sentiment_label (scoped to the ticker), url.
    Returns [] on error.
    """
    try:
        data = _cached_get("NEWS_SENTIMENT", ticker, tickers=ticker, limit="10")
        feed = data.get("feed", [])
        results = []
        for item in feed[:10]:
            label = "N/A"
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    label = ts.get("ticker_sentiment_label", "N/A")
                    break
            results.append(
                {
                    "title": item.get("title", ""),
                    "published": item.get("time_published", ""),
                    "sentiment_score": item.get("overall_sentiment_score", 0),
                    "sentiment_label": label,
                    "url": item.get("url", ""),
                }
            )
        return results
    except Exception as exc:
        logger.warning("get_news_sentiment failed for %s: %s", ticker, exc)
        return []


def format_market_summary(ticker: str) -> str:
    """
    Compose a compact text block for insertion into the LLM prompt.
    Called by query_engine at query time.
    """
    parts = [f"Ticker: {ticker}"]

    quote = get_quote(ticker)
    if quote:
        parts.append(
            f"Price: ${quote['price']} | Change: {quote['change']} ({quote['change_percent']}) "
            f"| Prev Close: ${quote['previous_close']}"
        )

    fundamentals = get_fundamentals(ticker)
    if fundamentals:
        mc = fundamentals["market_cap"]
        try:
            mc_display = f"${float(mc) / 1e9:.2f}B"
        except (ValueError, TypeError):
            mc_display = str(mc)
        parts.append(
            f"Market Cap: {mc_display} | P/E: {fundamentals['pe_ratio']} | "
            f"Forward P/E: {fundamentals['forward_pe']} | EPS: {fundamentals['eps']} | "
            f"Dividend Yield: {fundamentals['dividend_yield']} | "
            f"52W High: ${fundamentals['week_52_high']} | 52W Low: ${fundamentals['week_52_low']}"
        )

    news = get_news_sentiment(ticker)
    if news:
        parts.append("\nRecent news headlines (with sentiment):")
        for item in news[:5]:
            parts.append(f"  [{item['sentiment_label']}] {item['title']}")

    return "\n".join(parts)
