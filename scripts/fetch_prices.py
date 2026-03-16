"""
Sniper Price Pipeline
Fetches prices, 20-day ADV, and latest news for all tickers.
Writes data/prices.json for use by Command Center (index.html) and Streamlit app.

Run daily via GitHub Actions: .github/workflows/update_prices.yml
Exit 0 on success, 1 on failure.
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yfinance as yf

# --- Ticker lists ---
ARSENAL    = ['VRT', 'ASTS', 'VST', 'MU', 'APH', 'ANET', 'TSM', 'ONDS', 'FN']
WATCHTOWER = ['PLTR', 'TSEM', 'NBIS', 'BWXT', 'KTOS']
BENCHMARKS = ['SOXX', 'VOO', 'QQQ', 'EWY', 'GLD']
MACRO      = ['^VIX', 'THB=X']
ALL        = ARSENAL + WATCHTOWER + BENCHMARKS + MACRO

# ADV tracking: Watchtower + active-SO Arsenal tickers (FN has SO-5 gate)
ADV_TICKERS = WATCHTOWER + ['FN']

# Jan 1, 2026 close — YTD reference prices
YTD_REF = {
    'SOXX': 313.69,
    'VOO':  628.30,
    'QQQ':  613.12,
    'EWY':  102.22,
}

OUT_PATH = Path(__file__).resolve().parents[1] / 'data' / 'prices.json'


def get_thb_zone(thb: float) -> str:
    """THB/USD zone per Sniper Doctrine."""
    if thb < 32.0:
        return 'A'
    elif thb <= 36.0:
        return 'B'
    return 'C'


def fetch_prices() -> dict:
    """Download prices for all tickers, return structured price dict."""
    raw = yf.download(ALL, period='2d', progress=False, auto_adjust=True)

    # raw.columns is MultiIndex (metric, ticker) for multiple symbols
    close_df = raw['Close']  # DataFrame: rows=dates, cols=tickers

    prices = {}
    for sym in ALL:
        try:
            series = close_df[sym].dropna() if sym in close_df.columns else None
            if series is None or series.empty:
                print(f"  WARN: no data for {sym}", file=sys.stderr)
                continue

            price = float(series.iloc[-1])
            entry = {'price': round(price, 4)}

            if len(series) >= 2:
                prev = float(series.iloc[-2])
                entry['prev_close'] = round(prev, 4)
                if prev != 0:
                    entry['change_pct'] = round((price - prev) / prev * 100, 2)

            if sym in YTD_REF and YTD_REF[sym] != 0:
                entry['ytd_pct'] = round((price - YTD_REF[sym]) / YTD_REF[sym] * 100, 2)

            prices[sym] = entry

        except Exception as exc:
            print(f"  WARN: failed to process {sym}: {exc}", file=sys.stderr)

    return prices


def fetch_adv(tickers: list) -> dict:
    """
    Fetch 20-day ADV and today's volume for Watchtower + key gated tickers.
    Returns per-ticker: adv_20d, today_vol, adv_ratio, gate_met (>=1.5x).
    """
    adv = {}
    try:
        raw = yf.download(tickers, period='32d', progress=False, auto_adjust=True)
        vol_df = raw['Volume'] if 'Volume' in raw else None
        if vol_df is None:
            return adv

        for sym in tickers:
            try:
                col = vol_df[sym] if sym in vol_df.columns else None
                if col is None:
                    continue
                series = col.dropna()
                if len(series) < 2:
                    continue

                # 20-day ADV = average of up to 20 prior days (exclude today)
                prior = series.iloc[:-1]
                adv_20d = float(prior.iloc[-20:].mean())
                today_vol = float(series.iloc[-1])
                ratio = round(today_vol / adv_20d, 2) if adv_20d > 0 else None

                adv[sym] = {
                    'adv_20d':   round(adv_20d),
                    'today_vol': round(today_vol),
                    'adv_ratio': ratio,
                    'gate_met':  ratio is not None and ratio >= 1.5,
                }
            except Exception as exc:
                print(f"  WARN: ADV failed for {sym}: {exc}", file=sys.stderr)

    except Exception as exc:
        print(f"  WARN: ADV batch fetch failed: {exc}", file=sys.stderr)

    return adv


def fetch_news(tickers: list) -> dict:
    """
    Fetch most recent news headline for each Watchtower ticker via yfinance.
    Handles both legacy and current yfinance news schema defensively.
    """
    news = {}
    for sym in tickers:
        try:
            items = yf.Ticker(sym).news or []
            if not items:
                continue
            item = items[0]

            # Handle both yfinance news schema versions
            title = (
                item.get('content', {}).get('title')
                or item.get('title', '')
            )
            if title:
                title = title[:120]

            publisher = (
                item.get('content', {}).get('provider', {}).get('displayName')
                or item.get('publisher', '')
            )

            pub_ts = item.get('providerPublishTime', 0)
            if pub_ts:
                pub_date = datetime.fromtimestamp(pub_ts, tz=timezone.utc).strftime('%Y-%m-%d')
            else:
                pub_date = (item.get('content', {}).get('pubDate', '') or '')[:10]

            if title:
                news[sym] = {
                    'headline':  title,
                    'publisher': publisher,
                    'date':      pub_date,
                }
        except Exception as exc:
            print(f"  WARN: news failed for {sym}: {exc}", file=sys.stderr)

    return news


def compute_gate(prices: dict) -> dict:
    vix = prices.get('^VIX', {}).get('price', 0.0)
    thb = prices.get('THB=X', {}).get('price', 0.0)
    return {
        'vix_freeze': vix >= 25.0,
        'vix':        vix,
        'thb_zone':   get_thb_zone(thb) if thb > 0 else 'B',
        'thb':        thb,
    }


def build_output(prices: dict, adv: dict, news: dict) -> dict:
    now_utc = datetime.now(timezone.utc)
    bkk     = now_utc + timedelta(hours=7)
    return {
        'updated':     now_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'updated_bkk': bkk.strftime('%Y-%m-%d %H:%M GMT+7'),
        'prices':      prices,
        'ytd_ref':     YTD_REF,
        'gate':        compute_gate(prices),
        'adv':         adv,
        'news':        news,
    }


def main() -> int:
    print(f"Fetching {len(ALL)} tickers...")
    try:
        prices = fetch_prices()
    except Exception as exc:
        print(f"ERROR: fetch failed: {exc}", file=sys.stderr)
        return 1

    if not prices:
        print("ERROR: zero prices fetched", file=sys.stderr)
        return 1

    print(f"Fetching ADV for {len(ADV_TICKERS)} tickers...")
    adv = fetch_adv(ADV_TICKERS)

    print(f"Fetching news for {len(WATCHTOWER)} Watchtower tickers...")
    news = fetch_news(WATCHTOWER)

    output = build_output(prices, adv, news)
    gate   = output['gate']

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"Written: {OUT_PATH}")
    print(f"Updated: {output['updated']}")
    print(f"VIX:     {gate['vix']} — {'FREEZE' if gate['vix_freeze'] else 'CLEAR'}")
    print(f"THB:     {gate['thb']} — Zone {gate['thb_zone']}")
    print(f"Prices:  {len(prices)}/{len(ALL)} tickers")
    print(f"ADV:     {len(adv)}/{len(ADV_TICKERS)} tickers")
    print(f"News:    {len(news)}/{len(WATCHTOWER)} tickers")

    # Log any gate-met signals
    for sym, v in adv.items():
        if v.get('gate_met'):
            print(f"  ✓ ADV GATE MET: {sym} ratio={v['adv_ratio']}× (vol {v['today_vol']:,})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
