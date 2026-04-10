"""Reddit sentiment data fetching via the public JSON API (no auth required)."""

import logging
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "stockmarket"]

BULLISH_KEYWORDS = {
    "buy", "calls", "moon", "rocket", "bull", "undervalued",
    "long", "bullish", "breakout", "accumulate",
}

BEARISH_KEYWORDS = {
    "sell", "puts", "crash", "bear", "overvalued",
    "short", "bearish", "dump", "avoid",
}

_HEADERS = {"User-Agent": "StockPilot/1.0 (stock analysis bot)"}
_REQUEST_TIMEOUT = 10
_SLEEP_BETWEEN_REQUESTS = 1.0


def _classify_sentiment(text: str) -> str:
    """Return 'BULLISH', 'BEARISH', or 'NEUTRAL' based on keyword presence."""
    lowered = text.lower()
    words = set(lowered.split())
    bullish_hits = words & BULLISH_KEYWORDS
    bearish_hits = words & BEARISH_KEYWORDS
    if bullish_hits and not bearish_hits:
        return "BULLISH"
    if bearish_hits and not bullish_hits:
        return "BEARISH"
    if len(bullish_hits) > len(bearish_hits):
        return "BULLISH"
    if len(bearish_hits) > len(bullish_hits):
        return "BEARISH"
    return "NEUTRAL"


def _fetch_subreddit_posts(subreddit: str, ticker: str, limit: int = 10) -> list[dict]:
    """Fetch recent posts mentioning ticker from a subreddit."""
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": ticker,
        "sort": "new",
        "t": "week",
        "limit": limit,
        "restrict_sr": True,
    }
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 429:
            logger.warning("Reddit rate limited on r/%s, skipping", subreddit)
            return []
        if resp.status_code != 200:
            logger.warning("Reddit r/%s returned %d", subreddit, resp.status_code)
            return []
        data = resp.json()
        children = data.get("data", {}).get("children", [])
        posts = []
        for child in children:
            post = child.get("data", {})
            posts.append({
                "title": post.get("title", ""),
                "selftext": (post.get("selftext") or "")[:200],
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "subreddit": subreddit,
                "created_utc": post.get("created_utc", 0),
                "url": f"https://reddit.com{post.get('permalink', '')}",
            })
        return posts
    except Exception as e:
        logger.warning("Reddit fetch failed for r/%s: %s", subreddit, e)
        return []


def get_reddit_sentiment(ticker: str, curr_date: str = None) -> str:
    """Fetch and analyze Reddit sentiment for a stock ticker.

    Scans r/wallstreetbets, r/stocks, r/investing, r/stockmarket.
    Returns a formatted report string.
    """
    all_posts = []
    for subreddit in SUBREDDITS:
        posts = _fetch_subreddit_posts(subreddit, ticker)
        all_posts.extend(posts)
        if subreddit != SUBREDDITS[-1]:
            time.sleep(_SLEEP_BETWEEN_REQUESTS)

    if not all_posts:
        return f"Reddit sentiment data: No recent posts found for {ticker} in the last 7 days."

    # Deduplicate by title
    seen_titles = set()
    unique_posts = []
    for post in all_posts:
        if post["title"] not in seen_titles:
            seen_titles.add(post["title"])
            unique_posts.append(post)

    # Sort by score descending
    unique_posts.sort(key=lambda p: -p["score"])

    # Classify sentiment
    sentiments = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    weighted_sentiments = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    for post in unique_posts:
        text = f"{post['title']} {post['selftext']}"
        sentiment = _classify_sentiment(text)
        post["sentiment"] = sentiment
        sentiments[sentiment] += 1
        weight = max(1, post["score"])
        weighted_sentiments[sentiment] += weight

    total_weighted = sum(weighted_sentiments.values()) or 1
    bullish_pct = weighted_sentiments["BULLISH"] / total_weighted * 100
    bearish_pct = weighted_sentiments["BEARISH"] / total_weighted * 100
    neutral_pct = weighted_sentiments["NEUTRAL"] / total_weighted * 100

    if bullish_pct > bearish_pct + 10:
        overall = "BULLISH"
    elif bearish_pct > bullish_pct + 10:
        overall = "BEARISH"
    else:
        overall = "MIXED"

    lines = [
        f"=== Reddit Sentiment Analysis: {ticker} ===",
        f"Subreddits scanned: {', '.join(f'r/{s}' for s in SUBREDDITS)}",
        f"Period: Last 7 days",
        f"Total posts found: {len(unique_posts)}",
        f"Overall sentiment: {overall} ({bullish_pct:.0f}% bullish, {bearish_pct:.0f}% bearish, {neutral_pct:.0f}% neutral)",
        "",
        "Top Discussions:",
    ]

    for i, post in enumerate(unique_posts[:8], 1):
        lines.append(
            f"{i}. [{post['sentiment']}] [+{post['score']}] \"{post['title'][:80]}\" "
            f"(r/{post['subreddit']}, {post['num_comments']} comments)"
        )

    return "\n".join(lines)


def get_news(ticker: str, curr_date: str = None, *args, **kwargs) -> str:
    """Combined Reddit sentiment report — can replace social analyst's get_news."""
    return get_reddit_sentiment(ticker, curr_date)
