"""Finnhub vendor module for TradingAgents.

Provides news, fundamentals, insider data, and analyst recommendations
via the Finnhub REST API (https://finnhub.io/api/v1).

Auth: FINNHUB_API_KEY environment variable.
Rate limit: 60 calls/min — a 1-second sleep is inserted between each call.
"""

import logging
import os
import time
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
_SLEEP_BETWEEN_CALLS = 1.0
_REQUEST_TIMEOUT = 15


def _api_key() -> str:
    return os.environ.get("FINNHUB_API_KEY", "")


def _get(endpoint: str, params: dict) -> dict | list | None:
    """Perform a single authenticated GET request against the Finnhub API.

    Returns parsed JSON on success, None on any error.
    """
    key = _api_key()
    if not key:
        return None

    headers = {"X-Finnhub-Token": key}
    url = f"{FINNHUB_BASE_URL}{endpoint}"

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 429:
            logger.warning("Finnhub rate limit hit for %s, returning None", endpoint)
            return None
        if resp.status_code != 200:
            logger.warning("Finnhub %s returned HTTP %d", endpoint, resp.status_code)
            return None
        return resp.json()
    except Exception as exc:
        logger.warning("Finnhub request to %s failed: %s", endpoint, exc)
        return None


# ---------------------------------------------------------------------------
# Public vendor functions
# ---------------------------------------------------------------------------


def get_news(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
) -> str:
    """Fetch company-specific news from Finnhub plus an insider sentiment summary.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        start_date: Start date in YYYY-MM-DD format (default: 10 days ago).
        end_date: End date in YYYY-MM-DD format (default: today).

    Returns:
        Formatted text report suitable for LLM analysis.
    """
    if not _api_key():
        return "Finnhub data unavailable (no API key)"

    today = datetime.now()
    resolved_end = end_date or today.strftime("%Y-%m-%d")
    resolved_start = start_date or (today - timedelta(days=10)).strftime("%Y-%m-%d")

    news_data = _get(
        "/company-news",
        {"symbol": ticker.upper(), "from": resolved_start, "to": resolved_end},
    )
    time.sleep(_SLEEP_BETWEEN_CALLS)

    lines = [f"=== Finnhub Company News: {ticker.upper()} ==="]
    lines.append(f"Period: {resolved_start} to {resolved_end}\n")

    if not news_data:
        lines.append("No news articles found for the specified period.")
    else:
        for i, article in enumerate(news_data, 1):
            headline = article.get("headline") or "(no headline)"
            source = article.get("source") or "Unknown"
            summary = article.get("summary") or ""
            ts = article.get("datetime")
            date_str = (
                datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                if ts
                else "N/A"
            )
            lines.append(f"{i}. [{date_str}] {headline}")
            lines.append(f"   Source: {source}")
            if summary:
                lines.append(f"   Summary: {summary[:300]}")
            lines.append("")

    # Append insider sentiment summary
    lines.append("--- Insider Sentiment Summary ---")
    one_year_ago = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    sentiment_data = _get(
        "/stock/insider-sentiment",
        {"symbol": ticker.upper(), "from": one_year_ago, "to": resolved_end},
    )
    time.sleep(_SLEEP_BETWEEN_CALLS)

    if sentiment_data and isinstance(sentiment_data, dict):
        records = sentiment_data.get("data") or []
        if records:
            total_mspr = sum(r.get("mspr", 0) or 0 for r in records)
            avg_mspr = total_mspr / len(records)
            direction = "bullish (net buying)" if avg_mspr > 0 else "bearish (net selling)"
            lines.append(f"Monthly Share Purchase Ratio (MSPR) avg over {len(records)} months: {avg_mspr:.4f}")
            lines.append(f"Interpretation: Insiders are {direction}.")
        else:
            lines.append("No insider sentiment data available.")
    else:
        lines.append("Insider sentiment data unavailable.")

    return "\n".join(lines)


def get_global_news(
    curr_date: str = None,
    look_back_days: int = 7,
    limit: int = 5,
) -> str:
    """Fetch general market news from Finnhub.

    Args:
        curr_date: Reference date in YYYY-MM-DD format (default: today).
        look_back_days: Days of news to surface in the report (filters by datetime).
        limit: Maximum number of articles to include in the report.

    Returns:
        Formatted text report suitable for LLM analysis.
    """
    if not _api_key():
        return "Finnhub data unavailable (no API key)"

    today = datetime.now()
    resolved_date = curr_date or today.strftime("%Y-%m-%d")
    cutoff_dt = datetime.strptime(resolved_date, "%Y-%m-%d") - timedelta(days=look_back_days)

    news_data = _get("/news", {"category": "general"})
    time.sleep(_SLEEP_BETWEEN_CALLS)

    lines = [f"=== Finnhub Global Market News ==="]
    lines.append(f"As of {resolved_date} (last {look_back_days} days)\n")

    if not news_data:
        lines.append("No global news data available.")
        return "\n".join(lines)

    # Filter to look_back window and cap at limit
    filtered = [
        a for a in news_data
        if a.get("datetime") and datetime.fromtimestamp(a["datetime"]) >= cutoff_dt
    ]
    filtered = filtered[:limit]

    if not filtered:
        lines.append("No articles found within the specified look-back window.")
    else:
        for i, article in enumerate(filtered, 1):
            headline = article.get("headline") or "(no headline)"
            source = article.get("source") or "Unknown"
            summary = article.get("summary") or ""
            ts = article.get("datetime")
            date_str = (
                datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                if ts
                else "N/A"
            )
            lines.append(f"{i}. [{date_str}] {headline}")
            lines.append(f"   Source: {source}")
            if summary:
                lines.append(f"   Summary: {summary[:300]}")
            lines.append("")

    return "\n".join(lines)


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """Fetch key financial metrics and analyst recommendations from Finnhub.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        curr_date: Reference date (unused; provided for vendor signature compatibility).

    Returns:
        Formatted text report suitable for LLM analysis.
    """
    if not _api_key():
        return "Finnhub data unavailable (no API key)"

    metrics_data = _get("/stock/metric", {"symbol": ticker.upper(), "metric": "all"})
    time.sleep(_SLEEP_BETWEEN_CALLS)

    lines = [f"=== Finnhub Fundamentals: {ticker.upper()} ===\n"]

    if metrics_data and isinstance(metrics_data, dict):
        metric = metrics_data.get("metric") or {}

        field_map = [
            ("peBasicExclExtraTTM", "P/E Ratio (TTM)"),
            ("marketCapitalization", "Market Cap (M)"),
            ("52WeekHigh", "52-Week High"),
            ("52WeekLow", "52-Week Low"),
            ("dividendYieldIndicatedAnnual", "Dividend Yield (%)"),
            ("roeRfy", "Return on Equity (ROE)"),
            ("totalDebt/totalEquityAnnual", "Debt / Equity"),
            ("grossMarginTTM", "Gross Margin (%)"),
            ("netProfitMarginTTM", "Net Profit Margin (%)"),
            ("operatingMarginTTM", "Operating Margin (%)"),
            ("revenueGrowthTTMYoy", "Revenue Growth YoY (%)"),
            ("epsBasicExclExtraItemsTTM", "EPS (TTM)"),
            ("currentRatioAnnual", "Current Ratio"),
            ("bookValuePerShareAnnual", "Book Value Per Share"),
            ("beta", "Beta"),
        ]

        found_any = False
        for api_key_name, label in field_map:
            val = metric.get(api_key_name)
            if val is not None:
                lines.append(f"{label}: {val}")
                found_any = True

        if not found_any:
            lines.append("No metric data returned for this ticker.")
    else:
        lines.append("Metrics data unavailable.")

    # Analyst recommendations
    lines.append("\n--- Analyst Recommendations ---")
    rec_data = _get("/stock/recommendation", {"symbol": ticker.upper()})
    time.sleep(_SLEEP_BETWEEN_CALLS)

    if rec_data and isinstance(rec_data, list) and len(rec_data) > 0:
        latest = rec_data[0]
        strong_buy = latest.get("strongBuy", 0) or 0
        buy = latest.get("buy", 0) or 0
        hold = latest.get("hold", 0) or 0
        sell = latest.get("sell", 0) or 0
        strong_sell = latest.get("strongSell", 0) or 0
        period = latest.get("period", "N/A")

        total_buy = strong_buy + buy
        total_sell = sell + strong_sell

        lines.append(f"Period: {period}")
        lines.append(
            f"Consensus: {total_buy} Buy, {hold} Hold, {total_sell} Sell"
            f" (Strong Buy: {strong_buy}, Strong Sell: {strong_sell})"
        )

        total = total_buy + hold + total_sell
        if total > 0:
            if total_buy / total > 0.6:
                consensus_label = "BULLISH"
            elif total_sell / total > 0.4:
                consensus_label = "BEARISH"
            else:
                consensus_label = "NEUTRAL"
            lines.append(f"Overall Analyst Consensus: {consensus_label}")
    else:
        lines.append("No analyst recommendation data available.")

    return "\n".join(lines)


def get_insider_transactions(ticker: str) -> str:
    """Fetch insider sentiment and recent insider transactions from Finnhub.

    MSPR (Monthly Share Purchase Ratio):
        Positive -> insiders are net buyers (bullish signal).
        Negative -> insiders are net sellers (bearish signal).

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").

    Returns:
        Formatted text report suitable for LLM analysis.
    """
    if not _api_key():
        return "Finnhub data unavailable (no API key)"

    today = datetime.now()
    one_year_ago = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    lines = [f"=== Finnhub Insider Transactions: {ticker.upper()} ===\n"]

    # --- MSPR sentiment ---
    lines.append("-- Monthly Share Purchase Ratio (MSPR) --")
    sentiment_data = _get(
        "/stock/insider-sentiment",
        {"symbol": ticker.upper(), "from": one_year_ago, "to": today_str},
    )
    time.sleep(_SLEEP_BETWEEN_CALLS)

    if sentiment_data and isinstance(sentiment_data, dict):
        records = sentiment_data.get("data") or []
        if records:
            lines.append(f"{'Month':<12} {'MSPR':>10} {'Change':>12} {'Signal'}")
            lines.append("-" * 45)
            for record in records:
                month = record.get("month", "?")
                year = record.get("year", "?")
                mspr = record.get("mspr")
                change = record.get("change")
                mspr_str = f"{mspr:.4f}" if mspr is not None else "N/A"
                change_str = f"{change:+,}" if change is not None else "N/A"
                signal = "BULLISH" if (mspr or 0) > 0 else "BEARISH" if (mspr or 0) < 0 else "NEUTRAL"
                lines.append(f"{year}-{month:02d}    {mspr_str:>10} {change_str:>12} {signal}")

            avg_mspr = sum(r.get("mspr", 0) or 0 for r in records) / len(records)
            overall = "bullish (net buying)" if avg_mspr > 0 else "bearish (net selling)"
            lines.append(f"\n12-Month Avg MSPR: {avg_mspr:.4f} — Insiders are {overall}.")
        else:
            lines.append("No MSPR data available for the past year.")
    else:
        lines.append("Insider sentiment data unavailable.")

    # --- Recent transactions ---
    lines.append("\n-- Recent Insider Transactions --")
    tx_data = _get("/stock/insider-transactions", {"symbol": ticker.upper()})
    time.sleep(_SLEEP_BETWEEN_CALLS)

    if tx_data and isinstance(tx_data, dict):
        transactions = tx_data.get("data") or []
        if transactions:
            for tx in transactions[:10]:
                name = tx.get("name") or "Unknown"
                tx_code = tx.get("transactionCode") or ""
                shares = tx.get("share")
                price = tx.get("price")
                filing_date = tx.get("filingDate") or tx.get("transactionDate") or "N/A"
                shares_str = f"{shares:,}" if shares is not None else "N/A"
                price_str = f"${price:.2f}" if price is not None else "N/A"
                lines.append(
                    f"  {filing_date}  {name:<30} Code: {tx_code:<4}  "
                    f"Shares: {shares_str:>12}  Price: {price_str}"
                )
        else:
            lines.append("No recent insider transaction records found.")
    else:
        lines.append("Insider transaction data unavailable.")

    return "\n".join(lines)
