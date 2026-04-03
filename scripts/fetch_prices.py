"""
Sniper Price Pipeline — doctrine_ops v1.15.1
Fetches prices, technical indicators, ADV, and news for all tickers.
Writes data/prices.json for Command Center (index.html) and Streamlit app.

Run daily via GitHub Actions: .github/workflows/update_prices.yml
Exit 0 on success, 1 on failure.
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

import yfinance as yf
from ratelimit import limits, sleep_and_retry

# ── Ticker map (doctrine_ops v1.15.1 Section 0.4) ─────────────────────────────
ARSENAL          = ['VRT', 'ASTS', 'VST', 'MU', 'APH', 'ANET', 'TSM', 'ONDS', 'FN', 'COHR']
WATCHTOWER_CORE  = ['TSEM', 'BWXT', 'MOD', 'NBIS', 'FORM', 'ENTG', 'ONTO', 'LITE', 'QRVO', 'PLTR', 'KTOS', 'SKYT']
WATCHTOWER_SAT   = ['RKLB', 'SATL', 'PL']
SECTOR_ETFS      = ['SOXX', 'ITA', 'XLU']
BENCHMARKS       = ['VOO', 'QQQ', 'EWY', 'GLD']
SPY_TICKER       = 'SPY'
MACRO            = ['^VIX', 'THB=X']

# All tickers that need OHLCV history (SMA, 52w range, ADV, change5d)
PRICE_TICKERS = ARSENAL + WATCHTOWER_CORE + WATCHTOWER_SAT + SECTOR_ETFS + BENCHMARKS + [SPY_TICKER]
# Full fetch list
ALL = PRICE_TICKERS + MACRO

# Jan 1, 2026 close — YTD reference prices
YTD_REF = {
    'SOXX': 313.69,
    'VOO':  628.30,
    'QQQ':  613.12,
    'EWY':  102.22,
}

# VIX regime bands (doctrine_ops v1.15.1)
def get_regime(vix: float) -> str:
    """VIX regime bands per doctrine_core v1.12.1.
    30.0 exactly = ORANGE (within 25-30 band). RED = strictly above 30.
    """
    if vix < 22.0:
        return 'GREEN'
    elif vix < 25.0:
        return 'YELLOW'
    elif vix <= 30.0:    # 30.0 inclusive = ORANGE; >30 = RED
        return 'ORANGE'
    return 'RED'

# THB/USD zone per Sniper Doctrine
def get_thb_zone(thb: float) -> str:
    if thb < 32.0:
        return 'A'
    elif thb <= 36.0:
        return 'B'
    return 'C'

OUT_PATH = Path(__file__).resolve().parents[1] / 'data' / 'prices.json'

# ── Rate limiting for individual ticker API calls ─────────────────────────────
NEWS_CALLS_PER_PERIOD = 3   # max 3 news calls per second
NEWS_PERIOD_SECONDS   = 1

@sleep_and_retry
@limits(calls=NEWS_CALLS_PER_PERIOD, period=NEWS_PERIOD_SECONDS)
def _fetch_news_rate_limited(sym: str) -> list:
    """Fetch news for a single ticker, rate-limited."""
    items = yf.Ticker(sym).news or []
    return items


def _parse_news_item(item: dict, sym: str) -> dict | None:
    """Parse a yfinance news item into {title, url, date} format."""
    # Handle both yfinance news schema versions
    content = item.get('content', {})
    title = content.get('title') or item.get('title', '')
    if not title:
        return None
    title = title[:140]

    url = (
        content.get('canonicalUrl', {}).get('url')
        or content.get('clickThroughUrl', {}).get('url')
        or item.get('link', '')
        or ''
    )

    publisher = (
        content.get('provider', {}).get('displayName')
        or item.get('publisher', '')
    )

    pub_ts = item.get('providerPublishTime', 0)
    if pub_ts:
        pub_date = datetime.fromtimestamp(pub_ts, tz=timezone.utc).strftime('%Y-%m-%d')
    else:
        pub_date = (content.get('pubDate', '') or '')[:10]

    # Keep legacy headline/publisher keys for backward compat
    return {
        'title':     title,
        'headline':  title,      # backward compat
        'url':       url,
        'publisher': publisher,  # backward compat
        'date':      pub_date,
    }


def fetch_ohlcv_batch() -> tuple[dict, dict, dict]:
    """
    Batch-download ~250 days of OHLCV for all PRICE_TICKERS + MACRO.
    Returns (close_df, volume_df, high_df, low_df) as dicts of series.
    Also fetches 2-day data for macro tickers (VIX, THB=X).
    """
    import pandas as pd

    # Download historical data for price tickers (need 200+ days for SMA200)
    print(f"  Downloading OHLCV history for {len(PRICE_TICKERS)} tickers (250d)...")
    raw = yf.download(PRICE_TICKERS, period='250d', progress=False, auto_adjust=True)

    # Download macro tickers separately (short period — no SMA needed)
    print(f"  Downloading macro tickers {MACRO}...")
    raw_macro = yf.download(MACRO, period='5d', progress=False, auto_adjust=True)

    return raw, raw_macro


def _get_series(df, sym: str, col: str):
    """Safely extract a column series for a ticker from a downloaded DataFrame."""
    try:
        if hasattr(df.columns, 'levels'):
            # MultiIndex: (metric, ticker)
            if col in df.columns.get_level_values(0) and sym in df[col].columns:
                return df[col][sym].dropna()
        else:
            if col in df.columns:
                return df[col].dropna()
    except Exception:
        pass
    return None


def build_prices_from_history(raw, raw_macro) -> dict:
    """
    Process downloaded OHLCV data and return enriched per-ticker price dict.
    Computes: price, change, change_pct, change5d, adv, volume,
              high52w, low52w, sma50, sma200, prev_close, ytd_pct.
    """
    prices = {}

    for sym in PRICE_TICKERS:
        try:
            close = _get_series(raw, sym, 'Close')
            vol   = _get_series(raw, sym, 'Volume')
            high  = _get_series(raw, sym, 'High')
            low   = _get_series(raw, sym, 'Low')

            if close is None or len(close) < 2:
                print(f"  WARN: insufficient close data for {sym}", file=sys.stderr)
                continue

            price = float(close.iloc[-1])
            prev  = float(close.iloc[-2])

            # Daily change
            change_pct = round((price - prev) / prev * 100, 2) if prev != 0 else 0.0

            # 5-day change
            change5d = None
            if len(close) >= 6:
                p5 = float(close.iloc[-6])
                if p5 != 0:
                    change5d = round((price - p5) / p5 * 100, 2)

            # 52-week high/low (from up to 252 days of high/low data)
            high52w = round(float(high.iloc[-252:].max()), 4) if high is not None and len(high) >= 1 else None
            low52w  = round(float(low.iloc[-252:].min()),  4) if low  is not None and len(low)  >= 1 else None

            # SMA50 / SMA200
            sma50  = None
            sma200 = None
            if len(close) >= 50:
                sma50 = round(float(close.rolling(50).mean().iloc[-1]),  4)
            if len(close) >= 200:
                sma200 = round(float(close.rolling(200).mean().iloc[-1]), 4)

            # 20-day ADV (exclude today) and last session volume
            adv_20d    = None
            last_vol   = None
            if vol is not None and len(vol) >= 2:
                prior_vol = vol.iloc[:-1]
                adv_20d   = round(float(prior_vol.iloc[-20:].mean())) if len(prior_vol) >= 1 else None
                last_vol  = round(float(vol.iloc[-1]))

            # Market time — last trade date from OHLCV index
            # yfinance returns timezone-aware timestamps; convert to UTC ISO format
            market_time = None
            ts = close.index[-1]
            if hasattr(ts, 'strftime'):
                if hasattr(ts, 'tz') and ts.tz is not None:
                    market_time = ts.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
                else:
                    # No timezone info — format without Z suffix
                    market_time = ts.strftime('%Y-%m-%dT%H:%M:%S')

            entry = {
                'price':      round(price, 4),
                'prev_close': round(prev,  4),
                'change':     change_pct,
                'change_pct': change_pct,   # backward compat
                'change5d':   change5d,
                'adv':        adv_20d,
                'volume':     last_vol,
                'high52w':    high52w,
                'low52w':     low52w,
                'sma50':      sma50,
                'sma200':     sma200,
                'market_time': market_time,
            }

            if sym in YTD_REF and YTD_REF[sym] != 0:
                entry['ytd_pct'] = round((price - YTD_REF[sym]) / YTD_REF[sym] * 100, 2)

            prices[sym] = entry

        except Exception as exc:
            print(f"  WARN: failed to process {sym}: {exc}", file=sys.stderr)

    # Process macro tickers (VIX, THB=X) — simpler: just price + change
    for sym in MACRO:
        try:
            close = _get_series(raw_macro, sym, 'Close')
            if close is None or len(close) < 1:
                print(f"  WARN: no data for macro {sym}", file=sys.stderr)
                continue

            price = float(close.iloc[-1])
            entry = {'price': round(price, 4)}

            if len(close) >= 2:
                prev = float(close.iloc[-2])
                entry['prev_close'] = round(prev, 4)
                if prev != 0:
                    entry['change']     = round((price - prev) / prev * 100, 2)
                    entry['change_pct'] = entry['change']

            prices[sym] = entry

        except Exception as exc:
            print(f"  WARN: failed to process macro {sym}: {exc}", file=sys.stderr)

    return prices


def build_adv_section(prices: dict) -> dict:
    """
    Build backward-compatible top-level 'adv' dict from enriched per-ticker data.
    Includes Watchtower + key gated Arsenal tickers (ADV gate check: >=1.5x).
    """
    adv_tickers = list(set(WATCHTOWER_CORE + WATCHTOWER_SAT + ['FN']))
    adv = {}
    for sym in adv_tickers:
        entry = prices.get(sym, {})
        adv_20d   = entry.get('adv')
        today_vol = entry.get('volume')
        if adv_20d is None or today_vol is None:
            continue
        ratio = round(today_vol / adv_20d, 2) if adv_20d > 0 else None
        adv[sym] = {
            'adv_20d':   adv_20d,
            'today_vol': today_vol,
            'adv_ratio': ratio,
            'gate_met':  ratio is not None and ratio >= 1.5,
        }
    return adv


def fetch_news_all(tickers: list) -> dict:
    """
    Fetch up to 3 recent news items for each ticker via yfinance (rate-limited).
    Returns {sym: [{title, url, date}, ...]} with max 3 items per ticker.
    Also maintains backward-compat top-level news dict {sym: {headline, publisher, date}}.
    """
    news_arrays  = {}   # new format: per-ticker array
    news_compat  = {}   # backward compat: single headline per ticker

    for sym in tickers:
        try:
            items = _fetch_news_rate_limited(sym)
            if not items:
                continue

            parsed = []
            for item in items[:5]:   # parse up to 5, keep best 3
                result = _parse_news_item(item, sym)
                if result:
                    parsed.append(result)
                if len(parsed) >= 3:
                    break

            if parsed:
                news_arrays[sym] = parsed
                # backward compat: keep first item's headline/publisher/date
                news_compat[sym] = {
                    'headline':  parsed[0]['headline'],
                    'publisher': parsed[0]['publisher'],
                    'date':      parsed[0]['date'],
                }

        except Exception as exc:
            print(f"  WARN: news failed for {sym}: {exc}", file=sys.stderr)

    return news_arrays, news_compat


def compute_gate(prices: dict) -> dict:
    vix = prices.get('^VIX', {}).get('price', 0.0)
    thb = prices.get('THB=X', {}).get('price', 0.0)
    return {
        'vix_freeze': vix >= 25.0,
        'vix':        vix,
        'thb_zone':   get_thb_zone(thb) if thb > 0 else 'B',
        'thb':        thb,
    }


def build_output(prices: dict, adv: dict, news_arrays: dict, news_compat: dict) -> dict:
    now_utc = datetime.now(timezone.utc)
    bkk     = now_utc + timedelta(hours=7)

    vix    = prices.get('^VIX', {}).get('price', 0.0)
    thb    = prices.get('THB=X', {}).get('price', 0.0)
    regime = get_regime(vix)

    # SPY vs 200DMA for DC-15 regime check
    spy_entry = prices.get(SPY_TICKER, {})
    spy_price  = spy_entry.get('price', 0.0)
    spy_sma200 = spy_entry.get('sma200')
    spy_above_200dma = bool(spy_sma200 and spy_price > spy_sma200)

    # Inject per-ticker news arrays into prices dict
    enriched_prices = {}
    for sym, entry in prices.items():
        enriched = dict(entry)
        if sym in news_arrays:
            enriched['news'] = news_arrays[sym]
        enriched_prices[sym] = enriched

    timestamp_iso = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

    return {
        # New top-level metadata (doctrine_ops v1.15.1)
        'last_updated':      timestamp_iso,
        'fx_rate':           thb,
        'vix':               vix,
        'spy_above_200dma':  spy_above_200dma,
        'regime':            regime,
        # Backward-compat keys
        'updated':           timestamp_iso,
        'updated_bkk':       bkk.strftime('%Y-%m-%d %H:%M GMT+7'),
        'prices':            enriched_prices,
        'ytd_ref':           YTD_REF,
        'gate':              compute_gate(prices),
        'adv':               adv,
        'news':              news_compat,   # backward compat: {sym: {headline, publisher, date}}
    }


def main() -> int:
    print(f"Fetching data for {len(ALL)} tickers (doctrine_ops v1.15.1)...")

    # 1. Batch OHLCV download
    try:
        raw, raw_macro = fetch_ohlcv_batch()
    except Exception as exc:
        print(f"ERROR: OHLCV batch download failed: {exc}", file=sys.stderr)
        return 1

    # 2. Build enriched per-ticker price data
    try:
        prices = build_prices_from_history(raw, raw_macro)
    except Exception as exc:
        print(f"ERROR: price processing failed: {exc}", file=sys.stderr)
        return 1

    if not prices:
        print("ERROR: zero prices processed", file=sys.stderr)
        return 1

    # 3. Build ADV section (backward compat)
    adv = build_adv_section(prices)

    # 4. Fetch news for all tickers (rate-limited)
    news_tickers = ARSENAL + WATCHTOWER_CORE + WATCHTOWER_SAT
    print(f"Fetching news for {len(news_tickers)} tickers (rate-limited)...")
    news_arrays, news_compat = fetch_news_all(news_tickers)

    # 5. Build and write output
    output = build_output(prices, adv, news_arrays, news_compat)
    gate   = output['gate']

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))

    # Summary
    vix    = gate['vix']
    regime = output['regime']
    print(f"Written:  {OUT_PATH}")
    print(f"Updated:  {output['updated']}")
    print(f"VIX:      {vix} — {regime} ({'FREEZE' if gate['vix_freeze'] else 'CLEAR'})")
    print(f"THB:      {gate['thb']} — Zone {gate['thb_zone']}")
    print(f"SPY 200DMA: {'ABOVE' if output['spy_above_200dma'] else 'BELOW'}")
    print(f"Prices:   {len(prices)}/{len(ALL)} tickers")
    print(f"ADV:      {len(adv)} tickers enriched")
    print(f"News:     {len(news_arrays)}/{len(news_tickers)} tickers")

    # Log any ADV gate-met signals
    for sym, v in adv.items():
        if v.get('gate_met'):
            print(f"  ✓ ADV GATE MET: {sym} ratio={v['adv_ratio']}× (vol {v['today_vol']:,})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
