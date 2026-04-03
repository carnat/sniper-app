"""News & market intelligence — Finnhub integration, symbol resolution, watchtower."""

import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf


def get_finnhub_api_key(secrets_key=None):
    """Resolve Finnhub API key from provided secret or environment with placeholder filtering."""
    candidates = []
    if secrets_key:
        candidates.append(str(secrets_key).strip())
    candidates.append(str(os.getenv("FINNHUB_API_KEY", "")).strip())

    for candidate in candidates:
        if not candidate:
            continue
        low = candidate.lower()
        if "your_" in low or "placeholder" in low or "finnhub_api_key_here" in low:
            continue
        return candidate
    return ""

_get_finnhub_api_key = get_finnhub_api_key


def resolve_symbol_with_search(ticker):
    """Resolve ticker to a best-match Yahoo symbol + names using yfinance Search."""
    raw = str(ticker or "").strip().upper()
    if not raw:
        return {"input": "", "symbol": "", "shortname": "", "longname": "", "exchange": ""}

    stripped = raw.replace(".BK", "")
    fallback = {
        "input": raw,
        "symbol": raw,
        "shortname": "",
        "longname": "",
        "exchange": "",
    }

    try:
        search = yf.Search(raw, max_results=10, news_count=0)
        quotes = getattr(search, "quotes", []) or []
        if not quotes:
            if raw.endswith(".BK"):
                return {**fallback, "symbol": raw}
            return fallback

        def score_quote(q):
            symbol = str(q.get("symbol", "") or "").upper()
            exch = str(q.get("exchange", "") or q.get("exchDisp", "") or "").upper()
            score = 0
            if symbol == raw:
                score += 100
            if symbol == stripped:
                score += 90
            if symbol.replace(".BK", "") == stripped:
                score += 80
            if raw.endswith(".BK") and symbol.endswith(".BK"):
                score += 60
            if raw.endswith(".BK") and ("BKK" in exch or "TH" in exch or "SET" in exch):
                score += 40
            return score

        best = max(quotes, key=score_quote)
        best_symbol = str(best.get("symbol", "") or "").strip().upper() or raw
        return {
            "input": raw,
            "symbol": best_symbol,
            "shortname": str(best.get("shortname", "") or "").strip(),
            "longname": str(best.get("longname", "") or "").strip(),
            "exchange": str(best.get("exchange", "") or best.get("exchDisp", "") or "").strip(),
        }
    except Exception:
        return fallback

_resolve_symbol_with_search = resolve_symbol_with_search


def get_watchtower_market_snapshot():
    """Get compact market context strip for Watchtower."""
    snapshot = {
        "us_status": "Unknown",
        "us_message": "",
        "indices": {},
    }

    try:
        market_us = yf.Market("US")
        status = market_us.status if isinstance(market_us.status, dict) else {}
        snapshot["us_status"] = str(status.get("status", "Unknown") or "Unknown")
        snapshot["us_message"] = str(status.get("message", "") or "")
    except Exception:
        pass

    try:
        index_symbols = ["^GSPC", "^IXIC", "^DJI", "^SET.BK"]
        hist = yf.download(index_symbols, period="2d", progress=False)
        close_frame = hist.get("Close") if isinstance(hist, pd.DataFrame) else None
        if isinstance(close_frame, pd.DataFrame):
            for sym in index_symbols:
                series = close_frame.get(sym)
                if series is None:
                    continue
                vals = pd.Series(series).dropna()
                if len(vals) >= 2:
                    prev_val = float(vals.iloc[-2])
                    last_val = float(vals.iloc[-1])
                    pct = ((last_val - prev_val) / prev_val * 100.0) if prev_val != 0 else 0.0
                    snapshot["indices"][sym] = {"last": last_val, "pct": pct}
        elif isinstance(close_frame, pd.Series):
            vals = close_frame.dropna()
            if len(vals) >= 2:
                prev_val = float(vals.iloc[-2])
                last_val = float(vals.iloc[-1])
                pct = ((last_val - prev_val) / prev_val * 100.0) if prev_val != 0 else 0.0
                snapshot["indices"]["^GSPC"] = {"last": last_val, "pct": pct}
    except Exception:
        pass

    return snapshot

_get_watchtower_market_snapshot = get_watchtower_market_snapshot


def _extract_domain(url):
    """Extract clean domain from a URL."""
    try:
        if not url:
            return ""
        domain = str(url).split("//", 1)[-1].split("/", 1)[0].lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def fetch_news_for_ticker(ticker, finnhub_api_key, allowed_sources=None, allowed_domains=None):
    """Fetch latest news for a ticker using Finnhub company-news with relevance filtering.

    Parameters
    ----------
    ticker : str
    finnhub_api_key : str
    allowed_sources : set | None
    allowed_domains : set | None
    """
    if allowed_sources is None:
        allowed_sources = set()
    if allowed_domains is None:
        allowed_domains = set()

    try:
        raw_ticker = str(ticker or "").strip().upper()
        resolved_meta = resolve_symbol_with_search(raw_ticker)
        raw_ticker = str(resolved_meta.get("symbol", raw_ticker) or raw_ticker).strip().upper()
        search_ticker = raw_ticker.replace(".BK", "")
        if not search_ticker:
            return []

        company_name = str(resolved_meta.get("longname", "") or "").strip()
        company_alias = str(resolved_meta.get("shortname", "") or "").strip()
        try:
            yf_symbol = raw_ticker if raw_ticker else search_ticker
            yf_obj = yf.Ticker(yf_symbol)
            info = yf_obj.info
            if isinstance(info, dict):
                if not company_name:
                    company_name = str(info.get("longName", "") or "").strip()
                if not company_alias:
                    company_alias = str(info.get("shortName", "") or "").strip()
        except Exception:
            pass

        def is_relevant_article(article):
            title = str(article.get("title", "") or "")
            description = str(article.get("description", "") or "")
            content = str(article.get("content", "") or "")
            source = str((article.get("source") or {}).get("name", "") or "")
            combined = f"{title} {description} {content} {source}".upper()

            source_name_lc = source.strip().lower()
            article_url = str(article.get("url", "") or "").lower()
            article_domain = _extract_domain(article_url)
            is_allowed_source = bool(source_name_lc and source_name_lc in allowed_sources)
            is_allowed_domain = bool(article_domain and (article_domain in allowed_domains or any(article_domain.endswith(f".{domain}") for domain in allowed_domains)))
            if (allowed_sources or allowed_domains) and not (is_allowed_source or is_allowed_domain):
                return False

            ticker_upper = search_ticker.upper()
            ticker_variants = [
                ticker_upper,
                f"${ticker_upper}",
                f"{ticker_upper}.BK",
                f"({ticker_upper})",
            ]

            has_ticker = False
            for token in ticker_variants:
                if token and token in combined:
                    has_ticker = True
                    break

            company_variants = []
            if company_name:
                company_variants.append(company_name.upper())
            if company_alias:
                company_variants.append(company_alias.upper())
            has_company = any(token and token in combined for token in company_variants)

            if not (has_ticker or has_company):
                return False

            finance_keywords = [
                "STOCK", "SHARE", "EARNINGS", "REVENUE", "PROFIT", "LOSS", "GUIDANCE",
                "MARKET", "INVESTOR", "TRADING", "DIVIDEND", "ANALYST", "QUARTER", "FINANC"
            ]
            has_finance_context = any(keyword in combined for keyword in finance_keywords)
            return has_finance_context

        if finnhub_api_key:
            from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
            to_date = datetime.utcnow().strftime("%Y-%m-%d")

            symbol_candidates = []
            for candidate in [raw_ticker, search_ticker, str(resolved_meta.get("input", "") or "").strip().upper()]:
                clean = str(candidate or "").strip().upper()
                if clean and clean not in symbol_candidates:
                    symbol_candidates.append(clean)

            for candidate_symbol in symbol_candidates:
                try:
                    response = requests.get(
                        "https://finnhub.io/api/v1/company-news",
                        params={
                            "symbol": candidate_symbol,
                            "from": from_date,
                            "to": to_date,
                            "token": finnhub_api_key,
                        },
                        timeout=8,
                    )
                    if response.status_code != 200:
                        continue
                    payload = response.json() if response.content else []
                    if not isinstance(payload, list) or not payload:
                        continue

                    normalized_finnhub = []
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        url = str(item.get("url", "") or "").strip()
                        headline = str(item.get("headline", "") or "").strip()
                        summary = str(item.get("summary", "") or "").strip()
                        source_name = str(item.get("source", "Finnhub") or "Finnhub").strip()
                        published_at = None
                        try:
                            ts = item.get("datetime")
                            if ts:
                                published_at = datetime.utcfromtimestamp(int(ts)).isoformat() + "Z"
                        except Exception:
                            published_at = None

                        article = {
                            "title": headline,
                            "description": summary,
                            "content": summary,
                            "url": url,
                            "publishedAt": published_at,
                            "source": {"name": source_name},
                        }
                        if article["title"] and article["url"] and is_relevant_article(article):
                            normalized_finnhub.append(article)

                    if normalized_finnhub:
                        return normalized_finnhub[:8]
                except Exception:
                    continue
    except Exception:
        pass

    return []

_fetch_news_for_ticker = fetch_news_for_ticker


def format_news_timestamp(published_at):
    """Format published timestamp into relative time and exact local time."""
    try:
        parsed = pd.to_datetime(published_at, utc=True, errors='coerce')
        if pd.isna(parsed):
            return ("Unknown", "Unknown")

        local_tz = datetime.now().astimezone().tzinfo
        parsed_local = parsed.tz_convert(local_tz)
        now_local = pd.Timestamp.now(tz=local_tz)
        delta_seconds = int((now_local - parsed_local).total_seconds())
        if delta_seconds < 0:
            delta_seconds = 0

        if delta_seconds < 60:
            relative = "just now"
        elif delta_seconds < 3600:
            relative = f"{delta_seconds // 60}m ago"
        elif delta_seconds < 172800:
            relative = f"{delta_seconds // 3600}h ago"
        else:
            relative = f"{delta_seconds // 86400}d ago"

        exact = parsed_local.strftime("%Y-%m-%d %H:%M %Z")
        return (relative, exact)
    except Exception:
        return ("Unknown", "Unknown")


def get_earnings_dates(tickers):
    """Fetch upcoming earnings dates for tickers using yfinance."""
    earnings = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            if hasattr(stock, 'info') and 'earningsDate' in stock.info:
                earnings[ticker] = stock.info['earningsDate']
        except Exception:
            pass
    return earnings
