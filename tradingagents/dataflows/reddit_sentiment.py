"""Reddit sentiment data fetching via the public JSON API + LLM classification."""

import json
import logging
import os
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "stockmarket"]

_HEADERS = {"User-Agent": "StockPilot/1.0 (stock analysis bot)"}
_REQUEST_TIMEOUT = 10
_SLEEP_BETWEEN_REQUESTS = 1.0


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


def _classify_with_llm(ticker: str, posts: list[dict]) -> dict:
    """Use OpenAI to classify sentiment of Reddit posts in one batch call.

    Returns dict with keys: posts (list with sentiment added), overall, bullish_pct, bearish_pct.
    Falls back to NEUTRAL for all posts if LLM call fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or not posts:
        return _fallback_result(posts)

    # Build compact post list for LLM
    post_summaries = []
    for i, p in enumerate(posts[:15]):  # max 15 posts to keep token usage low
        post_summaries.append(f"{i+1}. [{p['score']} upvotes] {p['title'][:100]}")

    prompt = f"""Classify each Reddit post about the stock {ticker} as BULLISH, BEARISH, or NEUTRAL.
Consider the context: sarcasm, memes, hype, criticism, news sentiment.

Posts:
{chr(10).join(post_summaries)}

Respond with ONLY a JSON object:
{{"posts": [{{"id": 1, "sentiment": "BULLISH/BEARISH/NEUTRAL"}}], "overall": "BULLISH/BEARISH/NEUTRAL", "summary": "one sentence explaining the overall Reddit mood"}}"""

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4.1-nano",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.1,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("LLM sentiment call failed: %d", resp.status_code)
            return _fallback_result(posts)

        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Parse JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)

        # Apply sentiments to posts
        sentiment_map = {p["id"]: p["sentiment"] for p in result.get("posts", [])}
        for i, post in enumerate(posts[:15]):
            post["sentiment"] = sentiment_map.get(i + 1, "NEUTRAL")
        for post in posts[15:]:
            post["sentiment"] = "NEUTRAL"

        # Calculate weighted percentages
        weighted = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
        for post in posts:
            weight = max(1, post["score"])
            weighted[post.get("sentiment", "NEUTRAL")] = weighted.get(post.get("sentiment", "NEUTRAL"), 0) + weight
        total = sum(weighted.values()) or 1

        return {
            "posts": posts,
            "overall": result.get("overall", "NEUTRAL"),
            "summary": result.get("summary", ""),
            "bullish_pct": weighted["BULLISH"] / total * 100,
            "bearish_pct": weighted["BEARISH"] / total * 100,
            "neutral_pct": weighted["NEUTRAL"] / total * 100,
        }
    except Exception as e:
        logger.warning("LLM sentiment classification failed: %s", e)
        return _fallback_result(posts)


def _fallback_result(posts: list[dict]) -> dict:
    """Return NEUTRAL for all posts when LLM is unavailable."""
    for post in posts:
        post["sentiment"] = "NEUTRAL"
    return {
        "posts": posts,
        "overall": "NEUTRAL",
        "summary": "Sentiment classification unavailable",
        "bullish_pct": 0,
        "bearish_pct": 0,
        "neutral_pct": 100,
    }


def get_reddit_sentiment(ticker: str, curr_date: str = None) -> str:
    """Fetch and analyze Reddit sentiment for a stock ticker.

    Scans r/wallstreetbets, r/stocks, r/investing, r/stockmarket.
    Uses LLM to classify sentiment (falls back to NEUTRAL if unavailable).
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

    # Classify sentiment via LLM
    result = _classify_with_llm(ticker, unique_posts)

    lines = [
        f"=== Reddit Sentiment Analysis: {ticker} ===",
        f"Subreddits scanned: {', '.join(f'r/{s}' for s in SUBREDDITS)}",
        f"Period: Last 7 days",
        f"Total posts found: {len(unique_posts)}",
        f"Overall sentiment: {result['overall']} ({result['bullish_pct']:.0f}% bullish, {result['bearish_pct']:.0f}% bearish, {result['neutral_pct']:.0f}% neutral)",
    ]

    if result.get("summary"):
        lines.append(f"AI Summary: {result['summary']}")

    lines.append("")
    lines.append("Top Discussions:")

    for i, post in enumerate(unique_posts[:8], 1):
        lines.append(
            f"{i}. [{post.get('sentiment', 'NEUTRAL')}] [+{post['score']}] \"{post['title'][:80]}\" "
            f"(r/{post['subreddit']}, {post['num_comments']} comments)"
        )

    return "\n".join(lines)


def get_news(ticker: str, curr_date: str = None, *args, **kwargs) -> str:
    """Combined Reddit sentiment report — can replace social analyst's get_news."""
    return get_reddit_sentiment(ticker, curr_date)
