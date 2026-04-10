import logging
import os
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BRIDGEWISE_BASE_URL = "https://rest.bridgewise.com"

# Module-level cache: maps ticker (uppercase) -> company dict from Bridgewise
_company_cache: dict[str, dict] = {}


def search_company(ticker: str, token: str, base_url: str) -> Optional[dict]:
    """Search Bridgewise companies list for a matching ticker."""
    ticker_upper = ticker.upper()
    if ticker_upper in _company_cache:
        return _company_cache[ticker_upper]

    headers = {"Authorization": f"Bearer {token}"}
    for page in range(1, 20):
        try:
            resp = requests.get(
                f"{base_url}/companies",
                params={"page_size": 100, "page": page},
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning("Bridgewise companies page %d returned %d", page, resp.status_code)
                break
            data = resp.json()
            companies = data if isinstance(data, list) else data.get("results", data.get("data", []))
            if not companies:
                break
            for company in companies:
                cticker = (company.get("ticker") or "").upper()
                if cticker:
                    _company_cache[cticker] = company
                if cticker == ticker_upper:
                    return company
        except Exception as e:
            logger.warning("Bridgewise company search error page %d: %s", page, e)
            break
    return None


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """Fetch Bridgewise fundamental analysis for a ticker.

    Matches the yfinance vendor signature: (ticker, curr_date) -> str
    """
    token = os.environ.get("BRIDGEWISE_JWT", "")
    base_url = os.environ.get("BRIDGEWISE_BASE_URL", BRIDGEWISE_BASE_URL)

    if not token:
        return f"Bridgewise fundamental data unavailable for {ticker} (no JWT token)"

    company = search_company(ticker, token, base_url)
    if not company:
        return f"Bridgewise: company not found for ticker {ticker}"

    company_id = company.get("id") or company.get("company_id")
    if not company_id:
        return f"Bridgewise: no company ID for {ticker}"

    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(
            f"{base_url}/companies/{company_id}/fundamental-analysis",
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            return f"Bridgewise fundamental API returned {resp.status_code} for {ticker}"
        data = resp.json()
    except Exception as e:
        logger.warning("Bridgewise fundamentals fetch failed for %s: %s", ticker, e)
        return f"Bridgewise fundamental data unavailable for {ticker}: {e}"

    # Build readable report
    score = data.get("score", data.get("overall_score", "N/A"))
    rating = data.get("rating", data.get("recommendation", "N/A"))
    sector = company.get("sector", "N/A")
    industry = company.get("industry", company.get("sub_sector", "N/A"))
    name = company.get("name", company.get("company_name", ticker))

    lines = [
        f"=== Bridgewise Fundamental Analysis: {ticker} ({name}) ===",
        f"Overall Score: {score}/100",
        f"Rating: {rating}",
        f"Sector: {sector}",
        f"Industry: {industry}",
    ]

    # Add available metrics
    for key, label in [
        ("revenue_growth", "Revenue Growth"),
        ("profitability_score", "Profitability Score"),
        ("debt_level", "Debt Level"),
        ("valuation_score", "Valuation Score"),
        ("momentum_score", "Momentum Score"),
        ("dividend_yield", "Dividend Yield"),
        ("market_cap", "Market Cap"),
        ("pe_ratio", "P/E Ratio"),
    ]:
        val = data.get(key)
        if val is not None:
            lines.append(f"{label}: {val}")

    # Add any summary/commentary
    summary = data.get("summary", data.get("commentary", ""))
    if summary:
        lines.append(f"\nAnalyst Commentary: {summary[:500]}")

    return "\n".join(lines)
