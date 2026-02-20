import os
import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
import json
import sqlite3
import uuid
import math
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry

def get_app_version():
    """Read app version from VERSION file using SemVer format."""
    try:
        version_file = Path(__file__).with_name("VERSION")
        if version_file.exists():
            version_value = version_file.read_text(encoding="utf-8").strip()
            if version_value:
                return version_value
    except Exception:
        pass
    return "0.0.0"

APP_VERSION = get_app_version()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SNIPER COMMAND", layout="wide", page_icon="🎯")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    h1 { font-size: 2.5em; margin-bottom: 0.2em; letter-spacing: 2px; }
    h2 { margin-top: 1.5em; margin-bottom: 0.8em; color: #58a6ff; }
    .metric-card { background-color: #161b22; border-left: 3px solid #58a6ff; padding: 15px; border-radius: 6px; margin-bottom: 10px; }
    .stMetric { background-color: #161b22; border-radius: 6px; padding: 15px; border-left: 3px solid #58a6ff; }
    .positive { color: #3fb950; font-weight: bold; }
    .negative { color: #f85149; font-weight: bold; }
    table { border-collapse: collapse; width: 100%; table-layout: auto; }
    th { 
        background-color: #21262d; 
        padding: 8px 6px; 
        text-align: left; 
        font-weight: 600;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.2;
        font-size: 0.9em;
        width: fit-content;
    }
    td { 
        padding: 10px 8px; 
        border-bottom: 1px solid #21262d;
        white-space: nowrap;
        width: auto;
    }
    /* Compact percentage columns - Fund Day Gain %, Master Day Gain %, Master vs Fund %, P/L % */
    .stDataFrame th:nth-child(6), .stDataFrame th:nth-child(8), 
    .stDataFrame th:nth-child(9), .stDataFrame th:nth-child(12) {
        width: 55px !important;
        max-width: 55px !important;
        min-width: 55px !important;
        font-size: 0.65em !important;
        padding: 4px 2px !important;
        line-height: 1.0 !important;
        text-align: center !important;
        word-break: break-word !important;
        white-space: normal !important;
    }
    .stDataFrame td:nth-child(6), .stDataFrame td:nth-child(8),
    .stDataFrame td:nth-child(9), .stDataFrame td:nth-child(12) {
        width: 55px !important;
        max-width: 55px !important;
        min-width: 55px !important;
        white-space: nowrap !important;
        padding: 8px 2px !important;
        text-align: center !important;
        font-size: 0.9em !important;
    }
    /* Master column - also compact */
    .stDataFrame th:nth-child(7), .stDataFrame td:nth-child(7) {
        width: 60px !important;
        max-width: 60px !important;
        min-width: 60px !important;
        text-align: center !important;
        padding: 6px 2px !important;
    }
    .stDataFrame th:nth-child(7) {
        font-size: 0.75em !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    .dataframe { font-size: 0.95em; }
    .sidebar-label-muted {
        color: #9B9A97;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin: 0.5rem 0 0.2rem 0.5rem;
    }
    .sidebar-nav-section {
        color: #9B9A97;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin: 1rem 0 0.2rem 0.5rem;
    }
    .sidebar-nav-active {
        background: rgba(255, 255, 255, 0.055);
        color: #FFFFFF;
        border-radius: 6px;
        padding: 0.25rem 0.5rem;
        margin: 0.1rem 0;
        font-weight: 500;
        font-size: 0.875rem;
        line-height: 1.2;
        min-height: 1.75rem;
        display: flex;
        align-items: center;
        cursor: default;
    }
    /* --- ALL sidebar buttons: borderless, compact, uniform height --- */
    div[data-testid="stSidebar"] [data-testid="stButton"] {
        margin: 0 !important;
    }
    div[data-testid="stSidebar"] [data-testid="stButton"] > button {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        border-radius: 6px !important;
        padding: 0.25rem 0.5rem !important;
        min-height: 1.75rem !important;
        justify-content: flex-start !important;
        outline: none !important;
        margin: 0.1rem 0 !important;
        color: #9B9A97 !important;
        transition: background 20ms ease-in 0s !important;
    }
    div[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        background: rgba(255, 255, 255, 0.055) !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stSidebar"] [data-testid="stButton"] > button:focus,
    div[data-testid="stSidebar"] [data-testid="stButton"] > button:focus-visible,
    div[data-testid="stSidebar"] [data-testid="stButton"] > button:active {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        background: rgba(255, 255, 255, 0.055) !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stSidebar"] [data-testid="stButton"] > button p {
        font-size: 0.875rem !important;
        line-height: 1.2 !important;
        font-weight: 500 !important;
        margin: 0 !important;
        color: inherit !important;
    }
    .sidebar-status-row {
        display: flex;
        align-items: center;
        gap: 0.42rem;
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        color: #9B9A97;
    }
    .sidebar-status-dot {
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 999px;
        display: inline-block;
    }
    .sidebar-status-ok .sidebar-status-dot {
        background: #3fb950;
    }
    .sidebar-status-missing .sidebar-status-dot {
        background: #f85149;
    }
    .sidebar-status-ok {
        color: #9B9A97;
    }
    .sidebar-status-missing {
        color: #f85149;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 SNIPER OS")
st.markdown("**HYBRID INTEL COMMAND CENTER**")
st.caption(f"Version {APP_VERSION}")
st.markdown("#####")
st.caption("⚡ **US Markets:** Live (yfinance) | **Thai Stocks:** Live (yfinance) | **Thai Funds:** Live (SEC API)")
st.markdown("---")

# --- DATA INJECTION ---
# Load portfolio data from Streamlit secrets (if available) or use empty defaults
# This keeps your holdings private and out of Git history

def load_portfolio_from_secrets():
    """Load portfolio data from st.secrets with fallback to empty arrays"""
    try:
        # Try to load US portfolio from secrets
        us_portfolio = {
            "Ticker": list(st.secrets.get("us_portfolio", {}).get("Ticker", [])),
            "Shares": list(st.secrets.get("us_portfolio", {}).get("Shares", [])),
            "Avg_Cost": list(st.secrets.get("us_portfolio", {}).get("Avg_Cost", []))
        }
        
        # Try to load Thai stocks from secrets
        thai_stocks = {
            "Ticker": list(st.secrets.get("thai_stocks", {}).get("Ticker", [])),
            "Shares": list(st.secrets.get("thai_stocks", {}).get("Shares", [])),
            "Avg_Cost": list(st.secrets.get("thai_stocks", {}).get("Avg_Cost", []))
        }
        
        # Try to load vault portfolio from secrets
        vault_portfolio = []
        if "vault_portfolio" in st.secrets:
            vault_data = st.secrets["vault_portfolio"]
            # Handle both list and dict formats
            if isinstance(vault_data, (list, tuple)):
                vault_portfolio = [dict(fund) for fund in vault_data]
            
        return us_portfolio, thai_stocks, vault_portfolio
    except Exception as e:
        # If secrets are not configured, return empty portfolios
        return (
            {"Ticker": [], "Shares": [], "Avg_Cost": []},
            {"Ticker": [], "Shares": [], "Avg_Cost": []},
            []
        )

def get_transactions_file_path():
    """Local file path for persistent transaction history."""
    return Path(".streamlit") / "transactions.json"

def load_transaction_history():
    """Load transaction history from local file with safe fallback."""
    try:
        file_path = get_transactions_file_path()
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []

def save_transaction_history():
    """Persist transaction history to local file."""
    try:
        file_path = get_transactions_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(st.session_state.transaction_history, file, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_alert_state_file_path():
    """Local file path for persistent alert trigger state."""
    return Path(".streamlit") / "alert_state.json"

def load_alert_state():
    """Load alert trigger state from local file."""
    try:
        file_path = get_alert_state_file_path()
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}

def save_alert_state():
    """Persist alert trigger state to local file."""
    try:
        file_path = get_alert_state_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(st.session_state.alert_state, file, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_analytics_snapshots_file_path():
    """Local file path for analytics snapshot history."""
    return Path(".streamlit") / "analytics_snapshots.json"

def load_analytics_snapshots():
    """Load analytics snapshots from local file."""
    try:
        file_path = get_analytics_snapshots_file_path()
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []

def save_analytics_snapshots():
    """Persist analytics snapshots to local file."""
    try:
        file_path = get_analytics_snapshots_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(st.session_state.analytics_snapshots, file, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_scenario_library_file_path():
    """Local file path for saved backtesting scenarios."""
    return Path(".streamlit") / "scenario_library.json"

def load_saved_scenarios():
    """Load saved scenario definitions from local file."""
    try:
        file_path = get_scenario_library_file_path()
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []

def save_saved_scenarios():
    """Persist saved scenarios to local file."""
    try:
        file_path = get_scenario_library_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(st.session_state.saved_scenarios, file, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_watchlists_file_path():
    """Local file path for persisted watchlists."""
    return Path(".streamlit") / "watchlists.json"

def load_watchlists():
    """Load watchlists from local file."""
    try:
        file_path = get_watchlists_file_path()
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    clean = {}
                    for name, symbols in data.items():
                        if not isinstance(name, str):
                            continue
                        if not isinstance(symbols, list):
                            continue
                        clean[name] = [str(s).strip().upper() for s in symbols if str(s).strip()]
                    return clean
    except Exception:
        pass
    return {}

def save_watchlists():
    """Persist watchlists to local file."""
    try:
        file_path = get_watchlists_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(st.session_state.watchlists, file, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_options_iv_history_file_path():
    """Local file path for observed ATM IV history used by IV rank/percentile."""
    return Path(".streamlit") / "options_iv_history.json"

def load_options_iv_history():
    """Load ATM IV history from local file."""
    try:
        file_path = get_options_iv_history_file_path()
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    clean = {}
                    for symbol, values in data.items():
                        if not isinstance(symbol, str) or not isinstance(values, list):
                            continue
                        clean_values = []
                        for row in values:
                            if not isinstance(row, dict):
                                continue
                            day = str(row.get("date", "")).strip()
                            iv_val = row.get("atm_iv", None)
                            try:
                                iv_float = float(iv_val)
                            except Exception:
                                continue
                            if day:
                                clean_values.append({"date": day, "atm_iv": iv_float})
                        clean[symbol] = clean_values[-400:]
                    return clean
    except Exception:
        pass
    return {}

def save_options_iv_history():
    """Persist ATM IV history to local file."""
    try:
        file_path = get_options_iv_history_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(st.session_state.options_iv_history, file, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_calendar_events_file_path():
    """Local file path for user-planned portfolio calendar events."""
    return Path(".streamlit") / "calendar_events.json"

def load_calendar_events():
    """Load user-planned calendar events from local file."""
    try:
        file_path = get_calendar_events_file_path()
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    clean_events = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        event_date = str(item.get("date", "")).strip()
                        if not event_date:
                            continue
                        clean_events.append({
                            "id": str(item.get("id", str(uuid.uuid4()))),
                            "date": event_date,
                            "event_type": str(item.get("event_type", "Reminder") or "Reminder"),
                            "title": str(item.get("title", "") or "").strip(),
                            "instrument": str(item.get("instrument", "") or "").strip().upper(),
                            "details": str(item.get("details", "") or "").strip(),
                            "status": str(item.get("status", "Planned") or "Planned"),
                        })
                    return clean_events[-1200:]
    except Exception:
        pass
    return []

def save_calendar_events():
    """Persist user-planned calendar events to local file."""
    try:
        file_path = get_calendar_events_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(st.session_state.calendar_events, file, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _record_atm_iv(symbol, atm_iv):
    """Record one ATM IV observation per local day for a symbol."""
    if atm_iv is None:
        return
    clean_symbol = _normalize_market_symbol(symbol)
    if not clean_symbol:
        return
    day_key = datetime.now().strftime("%Y-%m-%d")
    history = st.session_state.options_iv_history.get(clean_symbol, [])
    replaced = False
    for row in history:
        if str(row.get("date", "")) == day_key:
            row["atm_iv"] = float(atm_iv)
            replaced = True
            break
    if not replaced:
        history.append({"date": day_key, "atm_iv": float(atm_iv)})
    st.session_state.options_iv_history[clean_symbol] = history[-400:]
    save_options_iv_history()

def _compute_iv_rank_percentile(symbol, current_atm_iv):
    """Compute IV rank and percentile from locally observed ATM IV history."""
    if current_atm_iv is None:
        return None, None, 0
    clean_symbol = _normalize_market_symbol(symbol)
    history = st.session_state.options_iv_history.get(clean_symbol, [])
    values = [float(r.get("atm_iv")) for r in history if r.get("atm_iv") is not None]
    if len(values) < 2:
        return None, None, len(values)

    min_iv = min(values)
    max_iv = max(values)
    iv_rank = None
    if max_iv > min_iv:
        iv_rank = (float(current_atm_iv) - min_iv) / (max_iv - min_iv) * 100.0
        iv_rank = max(0.0, min(100.0, iv_rank))

    less_equal = len([v for v in values if v <= float(current_atm_iv)])
    iv_percentile = (less_equal / len(values)) * 100.0 if values else None
    return iv_rank, iv_percentile, len(values)

def _estimate_atm_iv(calls_df, puts_df, underlying_price):
    """Estimate ATM IV by averaging nearest-strike call/put implied volatility."""
    if underlying_price in [None, 0]:
        return None
    candidates = []
    for chain_df in [calls_df, puts_df]:
        if not isinstance(chain_df, pd.DataFrame) or len(chain_df) == 0:
            continue
        if "strike" not in chain_df.columns or "impliedVolatility" not in chain_df.columns:
            continue
        view = chain_df[["strike", "impliedVolatility"]].copy()
        view["strike"] = pd.to_numeric(view["strike"], errors="coerce")
        view["impliedVolatility"] = pd.to_numeric(view["impliedVolatility"], errors="coerce")
        view = view.dropna(subset=["strike", "impliedVolatility"])
        if len(view) == 0:
            continue
        view["dist"] = (view["strike"] - float(underlying_price)).abs()
        nearest = view.sort_values("dist").head(1)
        if len(nearest) > 0:
            candidates.append(float(nearest.iloc[0]["impliedVolatility"]))

    if not candidates:
        return None
    return float(sum(candidates) / len(candidates))

def _annotate_black_scholes_greeks(chain_df, option_type, underlying_price, expiry_text, risk_free_rate=0.04):
    """Annotate options rows with Black-Scholes greek estimates using implied volatility."""
    if not isinstance(chain_df, pd.DataFrame) or len(chain_df) == 0:
        return pd.DataFrame()
    if underlying_price in [None, 0]:
        return chain_df.copy()

    try:
        expiry_dt = datetime.strptime(str(expiry_text), "%Y-%m-%d")
    except Exception:
        return chain_df.copy()

    days_to_expiry = max((expiry_dt - datetime.now()).days, 1)
    time_years = max(days_to_expiry / 365.0, 1 / 365.0)
    normal = NormalDist()

    df = chain_df.copy()
    if "strike" not in df.columns or "impliedVolatility" not in df.columns:
        return df

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")

    deltas = []
    gammas = []
    thetas = []
    vegas = []

    for _, row in df.iterrows():
        strike = row.get("strike", None)
        sigma = row.get("impliedVolatility", None)
        if strike in [None, 0] or sigma in [None, 0] or pd.isna(strike) or pd.isna(sigma):
            deltas.append(None)
            gammas.append(None)
            thetas.append(None)
            vegas.append(None)
            continue

        s = float(underlying_price)
        k = float(strike)
        vol = max(float(sigma), 0.0001)
        t = time_years
        r = float(risk_free_rate)

        try:
            d1 = (math.log(s / k) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
            d2 = d1 - vol * math.sqrt(t)
        except Exception:
            deltas.append(None)
            gammas.append(None)
            thetas.append(None)
            vegas.append(None)
            continue

        pdf_d1 = normal.pdf(d1)
        cdf_d1 = normal.cdf(d1)
        cdf_d2 = normal.cdf(d2)

        gamma = pdf_d1 / (s * vol * math.sqrt(t))
        vega = s * pdf_d1 * math.sqrt(t) / 100.0

        if option_type.lower() == "call":
            delta = cdf_d1
            theta = (-(s * pdf_d1 * vol) / (2 * math.sqrt(t)) - r * k * math.exp(-r * t) * cdf_d2) / 365.0
        else:
            delta = cdf_d1 - 1
            theta = (-(s * pdf_d1 * vol) / (2 * math.sqrt(t)) + r * k * math.exp(-r * t) * normal.cdf(-d2)) / 365.0

        deltas.append(delta)
        gammas.append(gamma)
        thetas.append(theta)
        vegas.append(vega)

    df["Delta"] = deltas
    df["Gamma"] = gammas
    df["Theta/day"] = thetas
    df["Vega"] = vegas
    return df

def _normalize_market_symbol(raw_text):
    text = str(raw_text or "").strip().upper()
    if not text:
        return ""
    return text

def _get_base_symbol_universe():
    symbols = set()
    symbols.update([str(t).strip().upper() for t in us_portfolio.get("Ticker", []) if str(t).strip()])
    symbols.update([str(t).strip().upper() for t in thai_stocks.get("Ticker", []) if str(t).strip()])
    symbols.update([str(f.get("Master", "")).strip().upper() for f in vault_portfolio if str(f.get("Master", "")).strip() and str(f.get("Master", "")).strip().upper() != "N/A"])
    for watch_symbols in st.session_state.get("watchlists", {}).values():
        if isinstance(watch_symbols, list):
            symbols.update([str(t).strip().upper() for t in watch_symbols if str(t).strip()])
    return sorted(symbols)

@st.cache_data(ttl=300)
def fetch_quote_snapshot(symbols):
    """Fetch quote snapshot rows for a list of symbols."""
    rows = []
    for symbol in symbols:
        clean_symbol = _normalize_market_symbol(symbol)
        if not clean_symbol:
            continue
        try:
            ticker = yf.Ticker(clean_symbol)
            info = ticker.info if isinstance(ticker.info, dict) else {}
            fast = getattr(ticker, "fast_info", {}) or {}

            current_price = info.get("regularMarketPrice", None)
            if current_price is None:
                current_price = fast.get("lastPrice", None)

            prev_close = info.get("regularMarketPreviousClose", None)
            if prev_close is None:
                prev_close = fast.get("previousClose", None)

            change_pct = None
            if current_price not in [None, 0] and prev_close not in [None, 0]:
                change_pct = (float(current_price) - float(prev_close)) / float(prev_close) * 100.0

            rows.append({
                "Symbol": clean_symbol,
                "Price": float(current_price) if current_price not in [None, ""] else None,
                "Change %": float(change_pct) if change_pct is not None else None,
                "Market Cap": info.get("marketCap", None),
                "P/E": info.get("trailingPE", None),
                "Forward P/E": info.get("forwardPE", None),
                "Div Yield %": (float(info.get("dividendYield", 0.0)) * 100.0) if info.get("dividendYield", None) is not None else None,
                "Beta": info.get("beta", None),
                "52W High": info.get("fiftyTwoWeekHigh", None),
                "52W Low": info.get("fiftyTwoWeekLow", None),
                "Avg Vol": info.get("averageVolume", None),
                "Sector": info.get("sector", None),
            })
        except Exception:
            rows.append({"Symbol": clean_symbol})

    return pd.DataFrame(rows)

@st.cache_data(ttl=900)
def fetch_fundamental_snapshot(symbol):
    """Fetch Phase 2 fundamentals package for a symbol."""
    clean_symbol = _normalize_market_symbol(symbol)
    if not clean_symbol:
        return {
            "metrics": {},
            "trend_df": pd.DataFrame(),
            "income_df": pd.DataFrame(),
            "balance_df": pd.DataFrame(),
            "cashflow_df": pd.DataFrame(),
            "recommendations_df": pd.DataFrame(),
            "earnings_dates_df": pd.DataFrame(),
            "analyst_snapshot": {},
            "factor_scores": {},
        }

    def _to_float(value):
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _score_linear(value, min_val, max_val, invert=False):
        if value is None:
            return None
        if max_val == min_val:
            return None
        clamped = max(min(value, max_val), min_val)
        ratio = (clamped - min_val) / (max_val - min_val)
        if invert:
            ratio = 1.0 - ratio
        return ratio * 100.0

    try:
        ticker = yf.Ticker(clean_symbol)
        info = ticker.info if isinstance(ticker.info, dict) else {}

        revenue = _to_float(info.get("totalRevenue", None))
        net_income = _to_float(info.get("netIncomeToCommon", None))
        trailing_eps = _to_float(info.get("trailingEps", None))
        gross_margin_pct = _to_float(info.get("grossMargins", None))
        operating_margin_pct = _to_float(info.get("operatingMargins", None))
        roe_pct = _to_float(info.get("returnOnEquity", None))
        debt_to_equity = _to_float(info.get("debtToEquity", None))
        earnings_growth_pct = _to_float(info.get("earningsGrowth", None))
        revenue_growth_pct = _to_float(info.get("revenueGrowth", None))
        trailing_pe = _to_float(info.get("trailingPE", None))
        forward_pe = _to_float(info.get("forwardPE", None))
        price_to_book = _to_float(info.get("priceToBook", None))
        market_price = _to_float(info.get("regularMarketPrice", None))
        low_52w = _to_float(info.get("fiftyTwoWeekLow", None))
        high_52w = _to_float(info.get("fiftyTwoWeekHigh", None))

        metrics = {
            "Revenue": revenue,
            "Net Income": net_income,
            "EPS": trailing_eps,
            "Gross Margin %": (gross_margin_pct * 100.0) if gross_margin_pct is not None else None,
            "Operating Margin %": (operating_margin_pct * 100.0) if operating_margin_pct is not None else None,
            "ROE %": (roe_pct * 100.0) if roe_pct is not None else None,
            "Debt/Equity": debt_to_equity,
            "Earnings Growth %": (earnings_growth_pct * 100.0) if earnings_growth_pct is not None else None,
        }

        income_statement = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow_statement = ticker.cashflow
        recommendations = ticker.recommendations
        earnings_dates = ticker.earnings_dates

        trend_df = pd.DataFrame()
        if isinstance(income_statement, pd.DataFrame) and not income_statement.empty:
            working = income_statement.T.reset_index().rename(columns={"index": "Period"})
            selected_cols = ["Period"]
            if "Total Revenue" in working.columns:
                selected_cols.append("Total Revenue")
            if "Net Income" in working.columns:
                selected_cols.append("Net Income")
            if len(selected_cols) > 1:
                trend_df = working[selected_cols].copy()

        analyst_snapshot = {
            "Recommendation": str(info.get("recommendationKey", "")).replace("_", " ").title() if info.get("recommendationKey", None) else None,
            "Target Mean": _to_float(info.get("targetMeanPrice", None)),
            "Target Low": _to_float(info.get("targetLowPrice", None)),
            "Target High": _to_float(info.get("targetHighPrice", None)),
            "Analyst Opinions": _to_float(info.get("numberOfAnalystOpinions", None)),
        }

        rec_df = recommendations.copy() if isinstance(recommendations, pd.DataFrame) and not recommendations.empty else pd.DataFrame()
        if len(rec_df) > 0 and isinstance(rec_df.index, pd.DatetimeIndex):
            rec_df = rec_df.reset_index().rename(columns={"index": "Date"})

        earn_df = earnings_dates.copy() if isinstance(earnings_dates, pd.DataFrame) and not earnings_dates.empty else pd.DataFrame()
        if len(earn_df) > 0 and isinstance(earn_df.index, pd.DatetimeIndex):
            earn_df = earn_df.reset_index().rename(columns={"index": "Date"})

        statement_to_frame = lambda df: (df.T.reset_index().rename(columns={"index": "Period"}) if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame())
        income_df = statement_to_frame(income_statement)
        balance_df = statement_to_frame(balance_sheet)
        cashflow_df = statement_to_frame(cashflow_statement)

        quality_candidates = [
            _score_linear((gross_margin_pct * 100.0) if gross_margin_pct is not None else None, 20, 60),
            _score_linear((operating_margin_pct * 100.0) if operating_margin_pct is not None else None, 5, 35),
            _score_linear((roe_pct * 100.0) if roe_pct is not None else None, 5, 30),
            _score_linear(debt_to_equity, 0, 200, invert=True),
        ]
        growth_candidates = [
            _score_linear((earnings_growth_pct * 100.0) if earnings_growth_pct is not None else None, -10, 35),
            _score_linear((revenue_growth_pct * 100.0) if revenue_growth_pct is not None else None, -5, 25),
        ]
        value_candidates = [
            _score_linear(trailing_pe, 8, 45, invert=True),
            _score_linear(forward_pe, 8, 40, invert=True),
            _score_linear(price_to_book, 0.5, 8, invert=True),
        ]

        momentum_score = None
        if market_price is not None and low_52w is not None and high_52w is not None and high_52w > low_52w:
            momentum_score = ((market_price - low_52w) / (high_52w - low_52w)) * 100.0
            momentum_score = max(0.0, min(100.0, momentum_score))

        def _avg(items):
            valid = [v for v in items if v is not None]
            if not valid:
                return None
            return sum(valid) / len(valid)

        factor_scores = {
            "Quality": _avg(quality_candidates),
            "Growth": _avg(growth_candidates),
            "Value": _avg(value_candidates),
            "Momentum": momentum_score,
        }

        return {
            "metrics": metrics,
            "trend_df": trend_df,
            "income_df": income_df,
            "balance_df": balance_df,
            "cashflow_df": cashflow_df,
            "recommendations_df": rec_df,
            "earnings_dates_df": earn_df,
            "analyst_snapshot": analyst_snapshot,
            "factor_scores": factor_scores,
        }
    except Exception:
        return {
            "metrics": {},
            "trend_df": pd.DataFrame(),
            "income_df": pd.DataFrame(),
            "balance_df": pd.DataFrame(),
            "cashflow_df": pd.DataFrame(),
            "recommendations_df": pd.DataFrame(),
            "earnings_dates_df": pd.DataFrame(),
            "analyst_snapshot": {},
            "factor_scores": {},
        }

@st.cache_data(ttl=900)
def fetch_options_snapshot(symbol, expiration=None):
    """Fetch options chain scaffold data for a symbol (Phase 3 prep)."""
    clean_symbol = _normalize_market_symbol(symbol)
    if not clean_symbol:
        return [], pd.DataFrame(), pd.DataFrame()
    try:
        ticker = yf.Ticker(clean_symbol)
        expiries = list(getattr(ticker, "options", []) or [])
        if not expiries:
            return [], pd.DataFrame(), pd.DataFrame()

        selected_exp = expiration if expiration in expiries else expiries[0]
        chain = ticker.option_chain(selected_exp)
        calls = chain.calls if hasattr(chain, "calls") else pd.DataFrame()
        puts = chain.puts if hasattr(chain, "puts") else pd.DataFrame()
        return expiries, calls, puts
    except Exception:
        return [], pd.DataFrame(), pd.DataFrame()

def build_import_key(asset_class, action, symbol, fund_code, quantity, price, transaction_date):
    """Build deterministic key for idempotent CSV imports."""
    instrument = fund_code if fund_code else symbol
    return "|".join([
        str(asset_class).strip().upper(),
        str(action).strip().upper(),
        str(instrument).strip().upper(),
        f"{float(quantity):.8f}",
        f"{float(price):.8f}",
        str(transaction_date).strip()
    ])

def import_key_exists(import_key):
    """Check whether an imported transaction key already exists in history."""
    if not import_key:
        return False
    for txn in st.session_state.transaction_history:
        if txn.get("import_key") == import_key:
            return True
    return False

def get_lot_db_path():
    """Local SQLite file path for lot tracking."""
    return Path(".streamlit") / "portfolio_lots.db"

def init_lot_database():
    """Initialize FIFO lot-tracking database tables."""
    try:
        db_path = get_lot_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tax_lots (
                lot_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                currency TEXT NOT NULL,
                acquired_date TEXT NOT NULL,
                quantity_original REAL NOT NULL,
                quantity_remaining REAL NOT NULL,
                cost_per_unit REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'BUY'
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realized_lots (
                realized_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                currency TEXT NOT NULL,
                sale_date TEXT NOT NULL,
                lot_id TEXT NOT NULL,
                quantity_sold REAL NOT NULL,
                cost_per_unit REAL NOT NULL,
                sale_price REAL NOT NULL,
                realized_pl REAL NOT NULL,
                FOREIGN KEY (lot_id) REFERENCES tax_lots(lot_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lot_metadata (
                meta_key TEXT PRIMARY KEY,
                meta_value TEXT
            )
        ''')

        conn.commit()
        conn.close()
    except Exception:
        pass

def lot_record_buy(symbol, asset_type, currency, quantity, price, acquired_date=None, source='BUY'):
    """Insert a new buy lot for FIFO tracking."""
    try:
        if quantity <= 0:
            return
        if acquired_date is None:
            acquired_date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(get_lot_db_path())
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO tax_lots (
                lot_id, symbol, asset_type, currency, acquired_date,
                quantity_original, quantity_remaining, cost_per_unit, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                str(uuid.uuid4()), symbol, asset_type, currency, acquired_date,
                float(quantity), float(quantity), float(price), source
            )
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def lot_record_sell_fifo(symbol, asset_type, currency, quantity, sale_price, sale_date=None):
    """Consume open lots via FIFO and return realized P/L. Returns None if insufficient lots."""
    try:
        if quantity <= 0:
            return 0.0
        if sale_date is None:
            sale_date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(get_lot_db_path())
        cursor = conn.cursor()

        cursor.execute(
            '''
            SELECT lot_id, quantity_remaining, cost_per_unit
            FROM tax_lots
            WHERE symbol = ? AND asset_type = ? AND currency = ? AND quantity_remaining > 0
            ORDER BY acquired_date ASC, rowid ASC
            ''',
            (symbol, asset_type, currency)
        )
        lots = cursor.fetchall()

        total_available = sum(lot[1] for lot in lots)
        if total_available + 1e-9 < quantity:
            conn.close()
            return None

        to_sell = float(quantity)
        realized_total = 0.0
        for lot_id, qty_remaining, cost_per_unit in lots:
            if to_sell <= 0:
                break

            use_qty = min(qty_remaining, to_sell)
            realized_piece = (float(sale_price) - float(cost_per_unit)) * float(use_qty)
            realized_total += realized_piece

            new_qty = float(qty_remaining) - float(use_qty)
            cursor.execute(
                'UPDATE tax_lots SET quantity_remaining = ? WHERE lot_id = ?',
                (new_qty, lot_id)
            )

            cursor.execute(
                '''
                INSERT INTO realized_lots (
                    realized_id, symbol, asset_type, currency, sale_date,
                    lot_id, quantity_sold, cost_per_unit, sale_price, realized_pl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    str(uuid.uuid4()), symbol, asset_type, currency, sale_date,
                    lot_id, float(use_qty), float(cost_per_unit), float(sale_price), float(realized_piece)
                )
            )
            to_sell -= float(use_qty)

        conn.commit()
        conn.close()
        return realized_total
    except Exception:
        return None

def lot_record_sell_lifo(symbol, asset_type, currency, quantity, sale_price, sale_date=None):
    """Consume open lots via LIFO and return realized P/L. Returns None if insufficient lots."""
    try:
        if quantity <= 0:
            return 0.0
        if sale_date is None:
            sale_date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(get_lot_db_path())
        cursor = conn.cursor()

        cursor.execute(
            '''
            SELECT lot_id, quantity_remaining, cost_per_unit
            FROM tax_lots
            WHERE symbol = ? AND asset_type = ? AND currency = ? AND quantity_remaining > 0
            ORDER BY acquired_date DESC, rowid DESC
            ''',
            (symbol, asset_type, currency)
        )
        lots = cursor.fetchall()

        total_available = sum(lot[1] for lot in lots)
        if total_available + 1e-9 < quantity:
            conn.close()
            return None

        to_sell = float(quantity)
        realized_total = 0.0
        for lot_id, qty_remaining, cost_per_unit in lots:
            if to_sell <= 0:
                break

            use_qty = min(qty_remaining, to_sell)
            realized_piece = (float(sale_price) - float(cost_per_unit)) * float(use_qty)
            realized_total += realized_piece

            new_qty = float(qty_remaining) - float(use_qty)
            cursor.execute(
                'UPDATE tax_lots SET quantity_remaining = ? WHERE lot_id = ?',
                (new_qty, lot_id)
            )

            cursor.execute(
                '''
                INSERT INTO realized_lots (
                    realized_id, symbol, asset_type, currency, sale_date,
                    lot_id, quantity_sold, cost_per_unit, sale_price, realized_pl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    str(uuid.uuid4()), symbol, asset_type, currency, sale_date,
                    lot_id, float(use_qty), float(cost_per_unit), float(sale_price), float(realized_piece)
                )
            )
            to_sell -= float(use_qty)

        conn.commit()
        conn.close()
        return realized_total
    except Exception:
        return None

def lot_record_sell_average(symbol, asset_type, currency, quantity, sale_price, sale_date=None):
    """Consume open lots using average-cost method and return realized P/L."""
    try:
        if quantity <= 0:
            return 0.0
        if sale_date is None:
            sale_date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(get_lot_db_path())
        cursor = conn.cursor()

        cursor.execute(
            '''
            SELECT lot_id, quantity_remaining, cost_per_unit
            FROM tax_lots
            WHERE symbol = ? AND asset_type = ? AND currency = ? AND quantity_remaining > 0
            ORDER BY acquired_date ASC, rowid ASC
            ''',
            (symbol, asset_type, currency)
        )
        lots = cursor.fetchall()

        total_available = sum(float(lot[1]) for lot in lots)
        if total_available + 1e-9 < quantity:
            conn.close()
            return None

        if total_available <= 0:
            conn.close()
            return 0.0

        weighted_cost_sum = sum(float(lot[1]) * float(lot[2]) for lot in lots)
        avg_cost = weighted_cost_sum / total_available
        realized_total = (float(sale_price) - float(avg_cost)) * float(quantity)

        to_sell = float(quantity)
        for lot_id, qty_remaining, _ in lots:
            if to_sell <= 0:
                break

            use_qty = min(float(qty_remaining), to_sell)
            new_qty = float(qty_remaining) - use_qty
            cursor.execute(
                'UPDATE tax_lots SET quantity_remaining = ? WHERE lot_id = ?',
                (new_qty, lot_id)
            )

            realized_piece = (float(sale_price) - float(avg_cost)) * float(use_qty)
            cursor.execute(
                '''
                INSERT INTO realized_lots (
                    realized_id, symbol, asset_type, currency, sale_date,
                    lot_id, quantity_sold, cost_per_unit, sale_price, realized_pl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    str(uuid.uuid4()), symbol, asset_type, currency, sale_date,
                    lot_id, float(use_qty), float(avg_cost), float(sale_price), float(realized_piece)
                )
            )
            to_sell -= float(use_qty)

        conn.commit()
        conn.close()
        return realized_total
    except Exception:
        return None

def get_lot_method_for_asset(asset_type):
    """Resolve selected lot method policy by asset type."""
    policies = st.session_state.get("lot_method_policies", {})
    default_map = {
        "US Stock": "FIFO",
        "Thai Stock": "FIFO",
        "Mutual Fund": "AVERAGE",
    }
    return str(policies.get(asset_type, default_map.get(asset_type, "FIFO"))).upper()

def lot_apply_split(symbol, asset_type, currency, split_ratio):
    """Apply stock split ratio to open lots for a symbol."""
    try:
        ratio = float(split_ratio)
        if ratio <= 0:
            return False

        conn = sqlite3.connect(get_lot_db_path())
        cursor = conn.cursor()
        cursor.execute(
            '''
            UPDATE tax_lots
            SET
                quantity_original = quantity_original * ?,
                quantity_remaining = quantity_remaining * ?,
                cost_per_unit = cost_per_unit / ?
            WHERE symbol = ? AND asset_type = ? AND currency = ? AND quantity_remaining > 0
            ''',
            (ratio, ratio, ratio, symbol, asset_type, currency)
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def seed_opening_lots_from_portfolios(us_portfolio, thai_stocks, vault_portfolio):
    """Seed lot DB once from current starting positions (if DB is empty)."""
    try:
        conn = sqlite3.connect(get_lot_db_path())
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(1) FROM tax_lots")
        lot_count = cursor.fetchone()[0]
        if lot_count > 0:
            conn.close()
            return

        today = datetime.now().strftime("%Y-%m-%d")

        for ticker, shares, avg_cost in zip(us_portfolio.get('Ticker', []), us_portfolio.get('Shares', []), us_portfolio.get('Avg_Cost', [])):
            if float(shares) > 0:
                lot_record_buy(ticker, "US Stock", "USD", float(shares), float(avg_cost), acquired_date=today, source='OPENING')

        for ticker, shares, avg_cost in zip(thai_stocks.get('Ticker', []), thai_stocks.get('Shares', []), thai_stocks.get('Avg_Cost', [])):
            if float(shares) > 0:
                lot_record_buy(ticker, "Thai Stock", "THB", float(shares), float(avg_cost), acquired_date=today, source='OPENING')

        for fund in vault_portfolio:
            units = float(fund.get('Units', 0))
            cost = float(fund.get('Cost', 0))
            code = fund.get('Code', '')
            if units > 0 and code:
                lot_record_buy(code, "Mutual Fund", "THB", units, cost, acquired_date=today, source='OPENING')

        cursor.execute(
            "INSERT OR REPLACE INTO lot_metadata (meta_key, meta_value) VALUES (?, ?)",
            ("opening_seeded_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def log_transaction(
    ticker,
    transaction_type,
    shares,
    price,
    asset_type="US Stock",
    avg_cost_at_time=0,
    transaction_date=None,
    import_key=None,
    total_override=None,
    notes=None,
    journal_meta=None,
    master=None,
):
    """Log a transaction to history for P/L tracking"""
    from datetime import datetime

    currency = "USD" if asset_type == "US Stock" else "THB"
    if transaction_date is None:
        transaction_date = datetime.now().strftime("%Y-%m-%d")

    realized_pl = 0
    lot_method_used = ""
    if transaction_type == "Buy":
        lot_record_buy(ticker, asset_type, currency, float(shares), float(price), acquired_date=transaction_date)
    elif transaction_type == "Sell":
        lot_method_used = get_lot_method_for_asset(asset_type)
        if lot_method_used == "LIFO":
            realized_result = lot_record_sell_lifo(ticker, asset_type, currency, float(shares), float(price), sale_date=transaction_date)
        elif lot_method_used == "AVERAGE":
            realized_result = lot_record_sell_average(ticker, asset_type, currency, float(shares), float(price), sale_date=transaction_date)
        else:
            realized_result = lot_record_sell_fifo(ticker, asset_type, currency, float(shares), float(price), sale_date=transaction_date)

        if realized_result is None:
            realized_pl = (price - avg_cost_at_time) * shares
        else:
            realized_pl = realized_result

    transaction_total = shares * price if total_override is None else float(total_override)
    transaction = {
        "date": f"{transaction_date} {datetime.now().strftime('%H:%M:%S')}",
        "ticker": ticker,
        "type": transaction_type,
        "shares": shares,
        "price": price,
        "total": transaction_total,
        "asset_type": asset_type,
        "currency": currency,
        "realized_pl": realized_pl,
    }

    if import_key:
        transaction["import_key"] = import_key
    if notes:
        transaction["notes"] = notes
    if lot_method_used:
        transaction["lot_method"] = lot_method_used
    if asset_type == "Mutual Fund":
        master_value = str(master or "").strip().upper()
        if master_value:
            transaction["master"] = master_value

    if isinstance(journal_meta, dict):
        allowed_journal_fields = {
            "strategy",
            "session",
            "direction",
            "followed_rules",
            "confidence",
            "risk_amount",
            "entry_window",
            "mental_state",
            "journal_note",
        }
        for key, value in journal_meta.items():
            if key not in allowed_journal_fields:
                continue
            if value is None:
                continue
            if isinstance(value, str) and str(value).strip() == "":
                continue
            transaction[key] = value

    risk_amount = transaction.get("risk_amount", None)
    if transaction_type == "Sell" and risk_amount not in [None, 0, 0.0]:
        try:
            transaction["r_multiple"] = float(realized_pl) / float(risk_amount)
        except Exception:
            pass
    
    st.session_state.transaction_history.append(transaction)
    save_transaction_history()

def get_realized_pl_summary():
    """Calculate total realized P/L from transaction history by currency"""
    realized_usd = 0.0
    realized_thb = 0.0
    for txn in st.session_state.transaction_history:
        if txn["type"] == "Sell":
            if txn.get("currency") == "USD":
                realized_usd += txn.get("realized_pl", 0)
            else:
                realized_thb += txn.get("realized_pl", 0)
    return realized_usd, realized_thb

def get_transaction_dataframe():
    """Convert transaction history to DataFrame"""
    if not st.session_state.transaction_history:
        return pd.DataFrame(columns=["Date", "Ticker", "Type", "Shares", "Price", "Total", "Asset Type", "Currency", "Realized P/L", "Lot Method"])
    
    df = pd.DataFrame(st.session_state.transaction_history)
    df = df.rename(columns={
        "date": "Date",
        "ticker": "Ticker",
        "type": "Type",
        "shares": "Shares",
        "price": "Price",
        "total": "Total",
        "asset_type": "Asset Type",
        "currency": "Currency",
        "realized_pl": "Realized P/L"
    })
    if "lot_method" not in df.columns:
        df["lot_method"] = ""
    optional_defaults = {
        "strategy": "",
        "session": "",
        "direction": "",
        "followed_rules": "",
        "confidence": "",
        "risk_amount": "",
        "entry_window": "",
        "mental_state": "",
        "journal_note": "",
        "r_multiple": "",
    }
    for col, default_value in optional_defaults.items():
        if col not in df.columns:
            df[col] = default_value
    df = df.rename(columns={"lot_method": "Lot Method"})
    df = df.rename(columns={
        "strategy": "Strategy",
        "session": "Session",
        "direction": "Direction",
        "followed_rules": "Followed Rules",
        "confidence": "Confidence",
        "risk_amount": "Risk Amount",
        "entry_window": "Entry Window",
        "mental_state": "Mental State",
        "journal_note": "Journal Note",
        "r_multiple": "R-Multiple",
    })
    return df

def hydrate_portfolios_from_transaction_history():
    """Rebuild current holdings from persisted Buy/Sell history when session portfolios are empty."""
    history = st.session_state.get("transaction_history", [])
    if not history:
        return

    has_existing_positions = (
        len(st.session_state.us_portfolio.get("Ticker", [])) > 0
        or len(st.session_state.thai_stocks.get("Ticker", [])) > 0
        or len(st.session_state.vault_portfolio) > 0
    )
    if has_existing_positions:
        return

    us_map = {}
    thai_map = {}
    fund_map = {}
    existing_master_map = {
        str(fund.get("Code", "")).strip().upper(): str(fund.get("Master", "N/A") or "N/A").strip().upper()
        for fund in st.session_state.get("vault_portfolio", [])
        if str(fund.get("Code", "")).strip()
    }

    for txn in history:
        txn_type = str(txn.get("type", "")).strip().title()
        asset_type = txn.get("asset_type", "US Stock")
        ticker = str(txn.get("ticker", "")).strip().upper()
        if not ticker:
            continue

        if asset_type == "US Stock":
            target = us_map
        elif asset_type == "Thai Stock":
            target = thai_map
        elif asset_type == "Mutual Fund":
            target = fund_map
        else:
            continue

        if txn_type == "Split":
            split_ratio = float(txn.get("shares", 0) or 0)
            if split_ratio <= 0 or ticker not in target:
                continue
            current_qty = float(target[ticker]["qty"])
            current_cost = float(target[ticker]["cost"])
            target[ticker] = {
                "qty": current_qty * split_ratio,
                "cost": (current_cost / split_ratio) if split_ratio > 0 else current_cost
            }
            continue

        if txn_type not in ["Buy", "Sell"]:
            continue

        shares = float(txn.get("shares", 0) or 0)
        price = float(txn.get("price", 0) or 0)
        if shares <= 0 or price <= 0:
            continue

        existing = target.get(ticker, {"qty": 0.0, "cost": 0.0})
        default_existing = {"qty": 0.0, "cost": 0.0}
        if target is fund_map:
            default_existing["master"] = existing_master_map.get(ticker, "N/A")
        existing = target.get(ticker, default_existing)
        qty = float(existing["qty"])
        avg_cost = float(existing["cost"])

        if txn_type == "Buy":
            total_cost = (qty * avg_cost) + (shares * price)
            new_qty = qty + shares
            if new_qty > 0:
                if target is fund_map:
                    txn_master = str(txn.get("master", "")).strip().upper()
                    existing_master = str(existing.get("master", "N/A") or "N/A").strip().upper()
                    resolved_master = txn_master if txn_master and txn_master != "N/A" else existing_master
                    target[ticker] = {"qty": new_qty, "cost": total_cost / new_qty, "master": resolved_master or "N/A"}
                else:
                    target[ticker] = {"qty": new_qty, "cost": total_cost / new_qty}
        else:
            new_qty = qty - shares
            if new_qty <= 1e-9:
                target.pop(ticker, None)
            else:
                if target is fund_map:
                    target[ticker] = {
                        "qty": new_qty,
                        "cost": avg_cost,
                        "master": str(existing.get("master", "N/A") or "N/A").strip().upper(),
                    }
                else:
                    target[ticker] = {"qty": new_qty, "cost": avg_cost}

    st.session_state.us_portfolio = {
        "Ticker": list(us_map.keys()),
        "Shares": [float(us_map[t]["qty"]) for t in us_map.keys()],
        "Avg_Cost": [float(us_map[t]["cost"]) for t in us_map.keys()]
    }

    st.session_state.thai_stocks = {
        "Ticker": list(thai_map.keys()),
        "Shares": [float(thai_map[t]["qty"]) for t in thai_map.keys()],
        "Avg_Cost": [float(thai_map[t]["cost"]) for t in thai_map.keys()]
    }

    st.session_state.vault_portfolio = [
        {
            "Code": code,
            "Units": float(fund_map[code]["qty"]),
            "Cost": float(fund_map[code]["cost"]),
            "Master": str(fund_map[code].get("master", existing_master_map.get(code, "N/A")) or "N/A").strip().upper()
        }
        for code in fund_map.keys()
    ]

# Initialize session state for portfolios if not exists
if 'us_portfolio' not in st.session_state:
    us_data, thai_data, vault_data = load_portfolio_from_secrets()
    st.session_state.us_portfolio = us_data
    st.session_state.thai_stocks = thai_data
    st.session_state.vault_portfolio = vault_data

# Initialize transaction history
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = load_transaction_history()

# Rebuild holdings from persisted history when session starts empty
hydrate_portfolios_from_transaction_history()

# Initialize alert state
if 'alert_state' not in st.session_state:
    st.session_state.alert_state = load_alert_state()

# Initialize lot method policies
if 'lot_method_policies' not in st.session_state:
    st.session_state.lot_method_policies = {
        "US Stock": "FIFO",
        "Thai Stock": "FIFO",
        "Mutual Fund": "AVERAGE",
    }

# Initialize analytics snapshots
if 'analytics_snapshots' not in st.session_state:
    st.session_state.analytics_snapshots = load_analytics_snapshots()

# Initialize saved scenario library
if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = load_saved_scenarios()

# Initialize watchlists
if 'watchlists' not in st.session_state:
    st.session_state.watchlists = load_watchlists()

# Initialize options IV history
if 'options_iv_history' not in st.session_state:
    st.session_state.options_iv_history = load_options_iv_history()

# Initialize user calendar events
if 'calendar_events' not in st.session_state:
    st.session_state.calendar_events = load_calendar_events()

init_lot_database()
seed_opening_lots_from_portfolios(st.session_state.us_portfolio, st.session_state.thai_stocks, st.session_state.vault_portfolio)

us_portfolio = st.session_state.us_portfolio
thai_stocks = st.session_state.thai_stocks
vault_portfolio = st.session_state.vault_portfolio

# Mapping: Real fund names -> SEC API abbreviations (for cases where they differ)
fund_api_mapping = {
    "SCBS&P500FUND(SSFA)": "SCBS&P500FUND(SSFA)",
    "SCBGOLDHE": "SCBGOLDHFUND",  # SCBGOLDHE is a class under SCBGOLDHFUND (M0856_2553)
    "SCB70SSF(SSFX)": "SCB70SSF",
    "SCBS&P500(E)": "SCBS&P500FUND(E)",
    "SCBS&P500-SSF": "SCBS&P500FUND(SSF)",
    "SCB70-SSFX": "SCB70SSF(SSFX)",
}

def _normalize_fund_token(text):
    """Normalize fund labels for resilient matching across import formats."""
    return (
        str(text or "")
        .upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )

def _resolve_fund_proj_id_and_class(fund_code, registry):
    """Resolve SEC proj_id + class suffix from imported fund code variants."""
    code_raw = str(fund_code or "").strip().upper()
    if not code_raw:
        return None, None

    api_code = fund_api_mapping.get(code_raw, code_raw)

    class_suffix = None
    base_candidate = api_code
    if '(' in api_code and api_code.endswith(')'):
        base_candidate = api_code[:api_code.index('(')]
        class_suffix = api_code[api_code.index('('):]
    elif '-' in api_code:
        left, right = api_code.rsplit('-', 1)
        right = right.strip().upper()
        if right in {'E', 'A', 'SSF', 'SSFE', 'SSFA', 'SSFX'}:
            base_candidate = left.strip()
            class_suffix = f"({right})"

    candidate_bases = [base_candidate]
    if base_candidate and not base_candidate.endswith("FUND"):
        candidate_bases.append(f"{base_candidate}FUND")
    if base_candidate and class_suffix and class_suffix in {"(SSFE)", "(SSFA)", "(SSFX)"} and not base_candidate.endswith("SSF"):
        candidate_bases.append(f"{base_candidate}SSF")

    # 1) Direct exact lookup
    for candidate in candidate_bases:
        if candidate in registry:
            return registry[candidate], class_suffix

    # 2) Normalized exact lookup
    registry_norm = {_normalize_fund_token(k): v for k, v in registry.items()}
    for candidate in candidate_bases:
        norm = _normalize_fund_token(candidate)
        if norm in registry_norm:
            return registry_norm[norm], class_suffix

    # 3) Fuzzy contains lookup for aliases like SCBS&P500 -> SCBS&P500FUND
    best_match = None
    best_score = 10**9
    for candidate in candidate_bases:
        cand_norm = _normalize_fund_token(candidate)
        if len(cand_norm) < 4:
            continue
        for reg_key, proj_id in registry.items():
            reg_norm = _normalize_fund_token(reg_key)
            if cand_norm in reg_norm or reg_norm in cand_norm:
                score = abs(len(reg_norm) - len(cand_norm))
                if score < best_score:
                    best_score = score
                    best_match = proj_id
    if best_match:
        return best_match, class_suffix

    return None, class_suffix

# --- INTELLIGENCE ENGINES ---

# Rate Limiter: 5 calls per 1 second (SEC API limit)
def get_sec_api_keys():
    """Load SEC API keys from Streamlit secrets (preferred) or environment variables."""
    keys = []

    try:
        sec_cfg = st.secrets.get("sec_api", {})
        primary = str(sec_cfg.get("primary_key", "")).strip()
        secondary = str(sec_cfg.get("secondary_key", "")).strip()
        if primary:
            keys.append(primary)
        if secondary and secondary != primary:
            keys.append(secondary)
    except Exception:
        pass

    env_primary = str(os.getenv("SEC_API_PRIMARY_KEY", "")).strip()
    env_secondary = str(os.getenv("SEC_API_SECONDARY_KEY", "")).strip()
    if env_primary and env_primary not in keys:
        keys.append(env_primary)
    if env_secondary and env_secondary not in keys:
        keys.append(env_secondary)

    return keys

@sleep_and_retry
@limits(calls=5, period=1)
def call_sec_api(url):
    """Rate-limited API call to SEC endpoints"""
    api_keys = get_sec_api_keys()
    if not api_keys:
        return None

    for api_key in api_keys:
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0',
                'Ocp-Apim-Subscription-Key': api_key,
            }
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return response
        except Exception:
            pass
    return None

_FUND_REGISTRY_CACHE_PATH = Path(".streamlit/fund_registry_cache.json")
_FUND_REGISTRY_CACHE_MAX_AGE = 86400  # 24 hours

def _load_disk_fund_registry():
    """Load fund registry from disk cache if fresh enough."""
    try:
        if _FUND_REGISTRY_CACHE_PATH.exists():
            data = json.loads(_FUND_REGISTRY_CACHE_PATH.read_text())
            saved_ts = data.get("ts", 0)
            if time.time() - saved_ts < _FUND_REGISTRY_CACHE_MAX_AGE:
                registry = data.get("registry", {})
                if registry:
                    return registry
    except Exception:
        pass
    return None

def _save_disk_fund_registry(registry):
    """Persist fund registry to disk for fast cold starts."""
    try:
        _FUND_REGISTRY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FUND_REGISTRY_CACHE_PATH.write_text(json.dumps({"ts": time.time(), "registry": registry}))
    except Exception:
        pass

@st.cache_data(ttl=3600)
def build_fund_registry():
    """
    Build complete fund registry by querying all AMCs.
    Uses a 24-hour disk cache so cold starts are instant.
    Returns: {"SCBNDQ(E)": "M0000_2553", ...}
    """
    # Fast path: disk cache
    cached = _load_disk_fund_registry()
    if cached:
        return cached

    fund_registry = {}
    
    try:
        # STEP 1: Get AMC list
        amc_url = "https://api.sec.or.th/FundFactsheet/fund/amc"
        r_amc = call_sec_api(amc_url)
        
        if r_amc is None:
            return fund_registry
        
        amc_list = json.loads(r_amc.content)
        
        # STEP 2: For each AMC, get all funds
        for amc in amc_list:
            unique_id = amc.get('unique_id')
            if not unique_id:
                continue
                
            fund_url = f"https://api.sec.or.th/FundFactsheet/fund/amc/{unique_id}"
            r_funds = call_sec_api(fund_url)
            
            if r_funds is not None:
                try:
                    funds_list = json.loads(r_funds.content)
                    for fund in funds_list:
                        abbr = fund.get('proj_abbr_name')
                        proj_id = fund.get('proj_id')
                        if abbr and proj_id:
                            fund_registry[abbr] = proj_id
                except Exception:
                    pass
    except Exception:
        pass
    
    # Persist to disk for next cold start
    if fund_registry:
        _save_disk_fund_registry(fund_registry)

    return fund_registry

@st.cache_data(ttl=3600)
def fetch_fund_nav_with_previous(proj_id, fund_class=None):
    """
    Fetch latest NAV and previous day NAV for a fund by proj_id using FundDailyInfo endpoint.
    Skips weekends to reduce unnecessary API calls.
    Returns tuple: (last_val, previous_val) for day gain calculation.
    """
    try:
        def extract_nav(nav_data, fund_class):
            """Helper to extract NAV from response data"""
            if isinstance(nav_data, list) and len(nav_data) > 0:
                if fund_class:
                    class_suffix = fund_class.strip('()')
                    for item in nav_data:
                        class_name = item.get('class_abbr_name', '')
                        if class_name.endswith(fund_class) or class_name.endswith(class_suffix):
                            last_val = item.get('last_val')
                            previous_val = item.get('previous_val')
                            if last_val:
                                return (last_val, previous_val)
                
                latest = nav_data[0]
                return (latest.get('last_val'), latest.get('previous_val'))
            elif isinstance(nav_data, dict):
                return (nav_data.get('last_val'), nav_data.get('previous_val'))
            return (None, None)
        
        # Build list of candidate dates, skipping weekends (Sat=5, Sun=6)
        candidate_dates = []
        now = datetime.now()
        days_back = 0
        while len(candidate_dates) < 7 and days_back < 14:
            d = now - timedelta(days=days_back)
            if d.weekday() < 5:  # Mon-Fri only
                candidate_dates.append(d.strftime("%Y-%m-%d"))
            days_back += 1

        found_dates = []
        
        for nav_date in candidate_dates:
            nav_url = f"https://api.sec.or.th/FundDailyInfo/{proj_id}/dailynav/{nav_date}"
            r = call_sec_api(nav_url)
            
            if r is not None and r.status_code == 200:
                nav_data = json.loads(r.content)
                last_val, previous_val = extract_nav(nav_data, fund_class)
                
                if last_val:
                    try:
                        last_price = float(last_val)
                        
                        if previous_val and str(previous_val).strip() and previous_val not in ['0', 0, '0.0', 0.0, '']:
                            prev_price = float(previous_val)
                            return (last_price, prev_price)
                        
                        found_dates.append((nav_date, last_price))
                        
                        if len(found_dates) >= 2:
                            return (found_dates[0][1], found_dates[1][1])
                        
                    except Exception:
                        pass
        
        if len(found_dates) == 1:
            return (found_dates[0][1], 0.0)
            
    except Exception:
        pass
    
    return (0.0, 0.0)

@st.cache_data(ttl=3600)
def fetch_fund_nav(proj_id, fund_class=None):
    """
    Fetch latest NAV for a fund by proj_id using FundDailyInfo endpoint.
    Format: /FundDailyInfo/{proj_id}/dailynav/{nav_date}
    NAV price is in the 'last_val' field.
    
    Args:
        proj_id: Fund project ID (e.g., "M0311_2564")
        fund_class: Optional fund class suffix (e.g., "(E)", "(SSF)", "(SSFE)")
    """
    last_val, _ = fetch_fund_nav_with_previous(proj_id, fund_class)
    return last_val


# --- MASTER CORRELATION ENGINE ---
@st.cache_data(ttl=300)
def _fetch_master_trends_cached(masters):
    symbols = tuple(sorted({str(m).strip().upper() for m in masters if str(m).strip() and str(m).strip().upper() != "N/A"}))
    if not symbols:
        return {}

    result = {symbol: 0.0 for symbol in symbols}
    try:
        history = yf.download(list(symbols), period="2d", progress=False)
        if isinstance(history, pd.DataFrame) and not history.empty:
            close_frame = history.get("Close")
            if isinstance(close_frame, pd.Series):
                if len(symbols) == 1:
                    closes = close_frame.dropna()
                    if len(closes) >= 2 and float(closes.iloc[-2]) != 0.0:
                        result[symbols[0]] = float(((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]) * 100.0)
            elif isinstance(close_frame, pd.DataFrame):
                for symbol in symbols:
                    series = close_frame.get(symbol)
                    if series is None:
                        continue
                    closes = pd.Series(series).dropna()
                    if len(closes) >= 2 and float(closes.iloc[-2]) != 0.0:
                        result[symbol] = float(((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]) * 100.0)
    except Exception:
        pass

    return result


def get_master_data(fund_list):
    """
    Fetches live percent-change for master ETFs/indices from yfinance.
    Uses Ticker info for real-time up/down percentage from market.
    Handles US tickers (VOO, QQQ) and Thai indices (^SET.BK).
    Returns dict: {"VOO": 0.24, "^SET.BK": -0.12}
    """
    masters = [f.get('Master') for f in fund_list if f.get('Master') and f.get('Master') != 'N/A']
    if not masters:
        return {}

    with st.spinner('🔮 Tracking Master ETFs...'):
        return _fetch_master_trends_cached(tuple(masters))


@st.cache_data(ttl=120)
def _fetch_latest_close_prices(tickers):
    if not tickers:
        return {}
    try:
        downloaded = yf.download(list(tickers), period="1d", progress=False)
        if downloaded is None or downloaded.empty:
            return {ticker: 0.0 for ticker in tickers}

        close_frame = downloaded.get("Close") if isinstance(downloaded, pd.DataFrame) else None
        if close_frame is None:
            return {ticker: 0.0 for ticker in tickers}

        if isinstance(close_frame, pd.Series):
            if len(tickers) == 1 and not close_frame.empty:
                return {tickers[0]: float(close_frame.iloc[-1])}
            return {ticker: 0.0 for ticker in tickers}

        latest = close_frame.iloc[-1] if not close_frame.empty else pd.Series(dtype=float)
        prices = {}
        for ticker in tickers:
            value = latest.get(ticker, 0.0)
            try:
                prices[ticker] = float(value)
            except Exception:
                prices[ticker] = 0.0
        return prices
    except Exception:
        return {ticker: 0.0 for ticker in tickers}

def get_stock_data(portfolio_dict):
    df = pd.DataFrame(portfolio_dict)
    
    # Handle empty portfolio - return empty dataframe with correct columns
    if len(df) == 0:
        return pd.DataFrame(columns=['Ticker', 'Shares', 'Avg_Cost', 'Live Price', 'Value', 'Cost Basis', 'P/L', 'P/L %'])
    
    tickers = df['Ticker'].tolist()
    with st.spinner('Scanning US/Thai Stocks...'):
        price_map = _fetch_latest_close_prices(tuple(tickers))
        current_prices = [float(price_map.get(t, 0.0) or 0.0) for t in tickers]
    df['Live Price'] = current_prices
    df['Value'] = df['Shares'] * df['Live Price']
    df['Cost Basis'] = df['Shares'] * df['Avg_Cost']
    df['P/L'] = df['Value'] - df['Cost Basis']
    df['P/L %'] = (df['P/L'] / df['Cost Basis']) * 100
    return df

# --- FUND DATA FETCHER (Using Registry + NAV Lookup) ---
def get_fund_nav_by_code(fund_code, registry):
    """
    Get NAV for a fund using the pre-built registry.
    fund_code format: "SCBNDQ(E)", "SCBS&P500(SSFA)", etc.
    Uses fund_api_mapping for real names that differ from SEC API abbreviations.
    Includes fallback mechanisms for funds with navigation issues.
    """
    proj_id, class_suffix = _resolve_fund_proj_id_and_class(fund_code, registry)
    if proj_id:
        nav = fetch_fund_nav(proj_id, class_suffix)
        if nav > 0:
            return nav

        nav = fetch_fund_nav(proj_id, None)
        if nav > 0:
            return nav

        for alt_class in ['(E)', '(SSF)', '(SSFE)', '(SSFA)', '(SSFX)', '(A)']:
            nav = fetch_fund_nav(proj_id, alt_class)
            if nav > 0:
                return nav
    
    return 0.0

def get_fund_data(fund_list):
    data = []
    bar = st.progress(0, text="Building Fund Registry & Fetching NAVs...")

    # Build registry once (cached / disk-cached)
    registry = build_fund_registry()

    # Fetch master ETF trends (batch yfinance call)
    master_trends = get_master_data(fund_list)

    # Pre-resolve all proj_ids before parallel fetch
    resolved = []
    for fund in fund_list:
        proj_id, class_suffix = _resolve_fund_proj_id_and_class(fund.get('Code', ''), registry)
        resolved.append((fund, proj_id, class_suffix))

    bar.progress(0.15, text="Fetching fund NAVs in parallel...")

    # Parallel NAV fetch for all funds at once
    nav_results = {}  # index -> (current_nav, prev_nav)

    def _fetch_nav(idx, proj_id, class_suffix):
        if proj_id:
            return idx, fetch_fund_nav_with_previous(proj_id, class_suffix)
        return idx, (0.0, 0.0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_fetch_nav, i, proj_id, class_suffix): i
            for i, (fund, proj_id, class_suffix) in enumerate(resolved)
        }
        done_count = 0
        total = len(futures)
        for future in as_completed(futures):
            idx, nav_pair = future.result()
            nav_results[idx] = nav_pair
            done_count += 1
            bar.progress(0.15 + 0.75 * done_count / max(total, 1), text=f"Fetched NAV {done_count}/{total}")

    # Assemble rows
    for i, (fund, proj_id, class_suffix) in enumerate(resolved):
        current_nav, prev_nav = nav_results.get(i, (0.0, 0.0))

        fund_day_gain = 0.0
        nav = current_nav
        if current_nav > 0 and prev_nav > 0:
            fund_day_gain = ((current_nav - prev_nav) / prev_nav) * 100

        # Fallback with class suffix variations if not found
        if nav == 0:
            nav = get_fund_nav_by_code(fund['Code'], registry)
        if nav == 0:
            nav = fund['Cost']

        row = fund.copy()
        row['Last Price'] = nav
        row['Previous Price'] = prev_nav if (prev_nav > 0 and prev_nav != nav) else None
        row['Fund Day Gain %'] = fund_day_gain

        master_ticker = fund.get('Master')
        if master_ticker and master_ticker != 'N/A':
            row['Master'] = master_ticker
            row['Master Day Gain %'] = master_trends.get(master_ticker, 0.0)
        else:
            row['Master'] = 'N/A'
            row['Master Day Gain %'] = 0.0

        data.append(row)

    bar.progress(1.0)
    bar.empty()
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        return pd.DataFrame(columns=['Code', 'Units', 'Cost', 'Last Price', 'Previous Price', 'Fund Day Gain %', 'Master', 'Master Day Gain %', 'Master vs Fund %', 'Cost Basis', 'Value', 'P/L', 'P/L %'])
    
    df['Value'] = df['Units'] * df['Last Price']
    df['Cost Basis'] = df['Units'] * df['Cost']
    df['P/L'] = df['Value'] - df['Cost Basis']
    df['P/L %'] = (df['P/L'] / df['Cost Basis']) * 100
    df['Master vs Fund %'] = df['Fund Day Gain %'] - df['Master Day Gain %']
    df = df[['Code', 'Units', 'Cost', 'Last Price', 'Previous Price', 'Fund Day Gain %', 'Master', 'Master Day Gain %', 'Master vs Fund %', 'Cost Basis', 'Value', 'P/L', 'P/L %']]
    
    return df

# --- NEWS & ALERTS ENGINE ---

def get_newsapi_key():
    """Resolve NewsAPI key from secrets or environment with placeholder filtering."""
    candidates = []
    try:
        candidates.append(str(st.secrets.get("news_alerts", {}).get("newsapi_key", "")).strip())
    except Exception:
        pass
    candidates.append(str(os.getenv("NEWSAPI_KEY", "")).strip())

    for candidate in candidates:
        if not candidate:
            continue
        low = candidate.lower()
        if "your_" in low or "placeholder" in low or "newsapi_key_here" in low:
            continue
        return candidate
    return ""

@st.cache_data(ttl=3600)
def fetch_news_for_ticker(ticker):
    """Fetch latest news for a ticker using NewsAPI (free tier)"""
    try:
        newsapi_key = get_newsapi_key()
        if not newsapi_key:
            return []
        
        # Remove .BK extension for Thai stocks
        search_ticker = str(ticker or "").replace(".BK", "").strip().upper()
        if not search_ticker:
            return []

        headers = {"X-Api-Key": newsapi_key}
        query_candidates = [
            search_ticker,
            f"{search_ticker} stock",
        ]

        for query_text in query_candidates:
            url = (
                "https://newsapi.org/v2/everything"
                f"?q={query_text}&sortBy=publishedAt&language=en&pageSize=8"
            )
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                continue
            payload = response.json() if response.content else {}
            articles = payload.get("articles", []) if isinstance(payload, dict) else []
            if articles:
                return articles
    except Exception:
        pass
    
    return []

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
    """Fetch upcoming earnings dates for tickers using yfinance"""
    earnings = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            if hasattr(stock, 'info') and 'earningsDate' in stock.info:
                earnings[ticker] = stock.info['earningsDate']
        except:
            pass
    return earnings

def check_price_alerts(portfolio_dict, current_prices, asset_type):
    """Check holdings against threshold with dedupe + cooldown state."""
    alerts = []
    threshold = st.secrets.get("news_alerts", {}).get("price_alert_threshold", 5)
    cooldown_minutes = st.secrets.get("news_alerts", {}).get("alerts_cooldown_minutes", 60)
    now = datetime.now()
    
    tickers = portfolio_dict.get("Ticker", [])
    avg_costs = portfolio_dict.get("Avg_Cost", [])
    currency_symbol = "$" if asset_type == "US Stock" else "฿"
    
    for ticker, avg_cost, current_price in zip(tickers, avg_costs, current_prices):
        if avg_cost is None or avg_cost == 0:
            continue

        pct_change = ((current_price - avg_cost) / avg_cost) * 100
        direction = "up" if pct_change > 0 else "down"
        state_key = f"{asset_type}|{ticker}"
        previous_state = st.session_state.alert_state.get(state_key, {})
        previously_triggered = previous_state.get("triggered", False)
        previous_direction = previous_state.get("direction")
        last_alert_at = previous_state.get("last_alert_at")

        cooldown_ok = True
        if last_alert_at:
            last_alert_dt = pd.to_datetime(last_alert_at, errors='coerce')
            if not pd.isna(last_alert_dt):
                elapsed_seconds = (now - last_alert_dt.to_pydatetime()).total_seconds()
                cooldown_ok = elapsed_seconds >= (cooldown_minutes * 60)

        in_threshold = abs(pct_change) >= threshold
        is_new = False

        if in_threshold:
            crossed_threshold = (not previously_triggered) or (previous_direction != direction)
            is_new = crossed_threshold and cooldown_ok

            st.session_state.alert_state[state_key] = {
                "triggered": True,
                "direction": direction,
                "last_alert_at": now.isoformat() if is_new else last_alert_at,
                "last_pct": pct_change
            }
        
            alerts.append({
                "ticker": ticker,
                "change_pct": pct_change,
                "change_type": "📈 UP" if pct_change > 0 else "📉 DOWN",
                "current_price": current_price,
                "price_at_cost": avg_cost,
                "asset_type": asset_type,
                "currency_symbol": currency_symbol,
                "is_new": is_new
            })
        else:
            st.session_state.alert_state[state_key] = {
                "triggered": False,
                "direction": None,
                "last_alert_at": last_alert_at,
                "last_pct": pct_change
            }

    save_alert_state()
    
    return alerts

def _normalize_column_name(name):
    return str(name).strip().lower().replace(" ", "").replace("_", "")

def _get_first_matching_column(df, aliases):
    normalized_map = {_normalize_column_name(col): col for col in df.columns}
    for alias in aliases:
        key = _normalize_column_name(alias)
        if key in normalized_map:
            return normalized_map[key]
    return None

def _normalize_asset_class(value):
    text = str(value).strip().lower()
    if text in ["us stock", "us", "stock", "equity", "us_equity"]:
        return "US Stock"
    if text in ["thai stock", "th stock", "thai", "thai_equity", "th_equity"]:
        return "Thai Stock"
    if text in ["mutual fund", "fund", "thai fund", "th fund", "thai_mutual_fund"]:
        return "Mutual Fund"
    return None

def _looks_thai_market_hint(currency_value, market_value):
    currency_text = str(currency_value).strip().upper() if currency_value is not None else ""
    market_text = str(market_value).strip().upper() if market_value is not None else ""
    return (
        currency_text == "THB"
        or "SET" in market_text
        or "BANGKOK" in market_text
        or "THAILAND" in market_text
        or "BK" in market_text
    )

def _parse_numeric_value(value):
    """Parse numeric values from Yahoo-style CSV fields (commas, currency symbols, parentheses)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "":
        return None

    negative = text.startswith("(") and text.endswith(")")
    cleaned = (
        text.replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("$", "")
        .replace("฿", "")
        .replace("THB", "")
        .replace("USD", "")
        .strip()
    )
    try:
        number = float(cleaned)
        return -number if negative else number
    except Exception:
        return None

def _parse_date_value(value):
    """Parse dates from Yahoo exports (supports YYYYMMDD, YYYYMMDD.0, and normal date strings)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if text == "":
        return None

    # Handle numeric-like Yahoo dates such as 20260108 or 20260108.0
    compact = text.replace(".0", "") if text.endswith(".0") else text
    if compact.isdigit() and len(compact) == 8:
        try:
            parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
            if not pd.isna(parsed):
                return parsed.strftime("%Y-%m-%d")
        except Exception:
            pass

    parsed = pd.to_datetime(text, errors='coerce')
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")

def parse_transactions_csv(uploaded_file, default_asset_class):
    """Parse Yahoo-style CSV into canonical transaction rows.

    Supports:
    - transaction export: action + quantity + price + date
    - holdings export: symbol + shares + avg cost/cost basis
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, [f"Unable to read CSV: {exc}"]

    symbol_col = _get_first_matching_column(df, ["symbol", "ticker", "code", "holding", "security"])
    fund_col = _get_first_matching_column(df, ["fund_code", "fundcode", "fund"])
    action_col = _get_first_matching_column(df, ["action", "type", "transaction_type", "transaction type"])
    yahoo_tx_type_col = _get_first_matching_column(df, ["transaction_type", "transaction type"])
    quantity_col = _get_first_matching_column(df, ["quantity", "shares", "units", "qty", "shares_owned", "shares owned"])
    price_col = _get_first_matching_column(df, ["price", "trade_price", "fill_price", "avg_price", "nav", "purchase_price", "purchase price", "average_cost", "avg_cost", "avg cost", "cost", "current_price", "current price"])
    cost_basis_col = _get_first_matching_column(df, ["cost_basis", "book_cost", "book cost", "total_cost", "cost basis"])
    date_col = _get_first_matching_column(df, ["trade_date", "date", "transaction_date"])
    asset_col = _get_first_matching_column(df, ["asset_class", "asset_type", "class"])
    currency_col = _get_first_matching_column(df, ["currency", "curr", "trading_currency", "trading currency"])
    market_col = _get_first_matching_column(df, ["market", "exchange", "country", "region"])
    master_col = _get_first_matching_column(df, ["master", "master_etf", "benchmark"])

    # Default behavior: auto-detect per row with US fallback unless user forces a class.
    fallback_asset_class = "US Stock" if default_asset_class == "Auto-detect" else default_asset_class

    is_transaction_mode = action_col is not None
    is_holdings_mode = (action_col is None and quantity_col is not None and (price_col is not None or cost_basis_col is not None))

    errors = []
    if not quantity_col:
        errors.append("Missing quantity column (expected: quantity/shares/units)")
    if not price_col and not cost_basis_col:
        errors.append("Missing price or cost basis column (expected: price/nav/avg_cost or cost_basis)")
    if not symbol_col and not fund_col:
        errors.append("Missing instrument column (expected symbol/ticker/code or fund_code)")
    if not is_transaction_mode and not is_holdings_mode:
        errors.append("Could not detect CSV mode. Provide either action column (transactions) or holdings fields (shares + avg cost/cost basis).")
    if errors:
        return None, errors

    canonical_rows = []
    row_errors = []

    for idx, row in df.iterrows():
        row_num = idx + 2

        # Skip Yahoo cash transactions (e.g., $$CASH_TX with DEPOSIT/WITHDRAWAL)
        symbol_hint = ""
        if symbol_col and pd.notna(row.get(symbol_col)):
            symbol_hint = str(row[symbol_col]).strip().upper()
        tx_type_hint = ""
        if yahoo_tx_type_col and pd.notna(row.get(yahoo_tx_type_col)):
            tx_type_hint = str(row[yahoo_tx_type_col]).strip().upper()
        if symbol_hint.startswith("$$CASH") or tx_type_hint in {"DEPOSIT", "WITHDRAWAL", "CASH", "DIVIDEND CASH"}:
            continue

        if is_transaction_mode:
            action_raw = str(row[action_col]).strip().upper()
            action_inferred = False
            if action_raw in ["BUY", "B"]:
                action = "Buy"
            elif action_raw in ["SELL", "S"]:
                action = "Sell"
            else:
                # Yahoo holdings rows can appear with empty Transaction Type.
                if action_raw in ["", "NAN", "NONE"]:
                    action = "Buy"
                    action_inferred = True
                else:
                    row_errors.append(f"Row {row_num}: Unsupported action '{action_raw}'")
                    continue
        else:
            # Holdings snapshot import: treat each row as opening BUY.
            action = "Buy"
            action_inferred = True

        quantity = _parse_numeric_value(row[quantity_col])
        if quantity is None or quantity <= 0:
            # Yahoo portfolio exports include many quote-only rows with no position data.
            # Skip inferred holding rows quietly; keep strict validation for explicit transactions.
            if action_inferred:
                continue
            row_errors.append(f"Row {row_num}: Invalid quantity")
            continue

        price = _parse_numeric_value(row[price_col]) if price_col else None
        if (price is None or price <= 0) and cost_basis_col:
            cost_basis = _parse_numeric_value(row[cost_basis_col])
            if cost_basis is not None and cost_basis > 0:
                price = cost_basis / quantity

        if price is None or price <= 0:
            if action_inferred:
                continue
            row_errors.append(f"Row {row_num}: Invalid price/cost basis")
            continue

        asset_class = fallback_asset_class
        if asset_col and pd.notna(row.get(asset_col)):
            mapped_asset = _normalize_asset_class(row[asset_col])
            if mapped_asset:
                asset_class = mapped_asset

        symbol = ""
        fund_code = ""
        if fund_col and pd.notna(row.get(fund_col)) and str(row[fund_col]).strip():
            fund_code = str(row[fund_col]).strip().upper()
            asset_class = "Mutual Fund"
        elif symbol_col and pd.notna(row.get(symbol_col)):
            symbol = str(row[symbol_col]).strip().upper()
        else:
            row_errors.append(f"Row {row_num}: Missing symbol/fund code")
            continue

        # Per-row inference for Yahoo exports (when asset class column is absent/ambiguous)
        currency_hint = row.get(currency_col) if currency_col else None
        market_hint = row.get(market_col) if market_col else None
        if not fund_code and asset_col is None:
            if symbol.endswith(".BK") or _looks_thai_market_hint(currency_hint, market_hint):
                asset_class = "Thai Stock"
            elif fallback_asset_class == "Mutual Fund":
                # Keep explicit user default for files that are all-fund and provided as symbol-like codes
                asset_class = "Mutual Fund"
            else:
                asset_class = fallback_asset_class

        if asset_class == "Thai Stock" and symbol and not symbol.endswith(".BK"):
            symbol = f"{symbol}.BK"

        date_value = datetime.now().strftime("%Y-%m-%d")
        if date_col and pd.notna(row.get(date_col)) and str(row[date_col]).strip():
            parsed_date = _parse_date_value(row[date_col])
            if parsed_date is None:
                row_errors.append(f"Row {row_num}: Invalid date '{row[date_col]}'")
                continue
            date_value = parsed_date

        master = "N/A"
        if master_col and pd.notna(row.get(master_col)) and str(row[master_col]).strip():
            master = str(row[master_col]).strip().upper()

        canonical_rows.append({
            "row_num": row_num,
            "asset_class": asset_class,
            "action": action,
            "symbol": symbol,
            "fund_code": fund_code,
            "quantity": quantity,
            "price": price,
            "transaction_date": date_value,
            "master": master,
            "import_key": build_import_key(asset_class, action, symbol, fund_code, quantity, price, date_value)
        })

    return pd.DataFrame(canonical_rows), row_errors

def get_csv_templates():
    """Return downloadable CSV templates for US, Thai stocks, and Thai funds."""
    us_template = pd.DataFrame([
        {"symbol": "AAPL", "action": "BUY", "quantity": 10, "price": 190.50, "date": "2026-01-15"},
        {"symbol": "AAPL", "action": "SELL", "quantity": 2, "price": 201.20, "date": "2026-02-01"}
    ]).to_csv(index=False)

    thai_stock_template = pd.DataFrame([
        {"symbol": "ADVANC.BK", "action": "BUY", "quantity": 50, "price": 220.00, "date": "2026-01-10"},
        {"symbol": "TISCO.BK", "action": "BUY", "quantity": 100, "price": 98.50, "date": "2026-01-11"}
    ]).to_csv(index=False)

    thai_fund_template = pd.DataFrame([
        {"fund_code": "SCBNDQ(E)", "action": "BUY", "quantity": 1000, "price": 13.50, "date": "2026-01-05", "master": "QQQ"},
        {"fund_code": "SCBNDQ(E)", "action": "SELL", "quantity": 100, "price": 13.95, "date": "2026-02-12", "master": "QQQ"}
    ]).to_csv(index=False)

    mixed_template = pd.DataFrame([
        {"asset_class": "US Stock", "symbol": "MSFT", "action": "BUY", "quantity": 3, "price": 420.00, "date": "2026-01-07"},
        {"asset_class": "Thai Stock", "symbol": "ADVANC", "action": "BUY", "quantity": 20, "price": 221.00, "date": "2026-01-12"},
        {"asset_class": "Mutual Fund", "fund_code": "SCBS&P500FUND(E)", "action": "BUY", "quantity": 200, "price": 38.20, "date": "2026-01-18", "master": "VOO"}
    ]).to_csv(index=False)

    return {
        "us": us_template,
        "thai_stock": thai_stock_template,
        "thai_fund": thai_fund_template,
        "mixed": mixed_template
    }

def simulate_import_preview(parsed_df):
    """Dry-run import simulation to preview import/skip/fail outcomes."""
    us_pos = {ticker: float(shares) for ticker, shares in zip(st.session_state.us_portfolio.get("Ticker", []), st.session_state.us_portfolio.get("Shares", []))}
    thai_pos = {ticker: float(shares) for ticker, shares in zip(st.session_state.thai_stocks.get("Ticker", []), st.session_state.thai_stocks.get("Shares", []))}
    fund_pos = {fund.get("Code"): float(fund.get("Units", 0)) for fund in st.session_state.vault_portfolio}

    preview_rows = []
    import_count = 0
    skip_count = 0
    fail_count = 0

    for _, row in parsed_df.iterrows():
        asset_class = row["asset_class"]
        action = row["action"]
        quantity = float(row["quantity"])
        import_key = row.get("import_key", "")
        symbol_or_code = row["fund_code"] if row["fund_code"] else row["symbol"]

        status = "Import"
        reason = ""

        if import_key_exists(import_key):
            status = "Skip"
            reason = "Duplicate transaction"
            skip_count += 1
        else:
            if asset_class == "US Stock":
                current = us_pos.get(row["symbol"], 0.0)
                if action == "Sell" and quantity > current:
                    status = "Fail"
                    reason = f"Insufficient shares ({current:.2f} available)"
                    fail_count += 1
                elif action == "Buy":
                    us_pos[row["symbol"]] = current + quantity
                    import_count += 1
                else:
                    new_qty = current - quantity
                    if new_qty <= 0:
                        us_pos.pop(row["symbol"], None)
                    else:
                        us_pos[row["symbol"]] = new_qty
                    import_count += 1

            elif asset_class == "Thai Stock":
                current = thai_pos.get(row["symbol"], 0.0)
                if action == "Sell" and quantity > current:
                    status = "Fail"
                    reason = f"Insufficient shares ({current:.2f} available)"
                    fail_count += 1
                elif action == "Buy":
                    thai_pos[row["symbol"]] = current + quantity
                    import_count += 1
                else:
                    new_qty = current - quantity
                    if new_qty <= 0:
                        thai_pos.pop(row["symbol"], None)
                    else:
                        thai_pos[row["symbol"]] = new_qty
                    import_count += 1

            else:
                current = fund_pos.get(row["fund_code"], 0.0)
                if action == "Sell" and quantity > current:
                    status = "Fail"
                    reason = f"Insufficient units ({current:.2f} available)"
                    fail_count += 1
                elif action == "Buy":
                    fund_pos[row["fund_code"]] = current + quantity
                    import_count += 1
                else:
                    new_qty = current - quantity
                    if new_qty <= 0:
                        fund_pos.pop(row["fund_code"], None)
                    else:
                        fund_pos[row["fund_code"]] = new_qty
                    import_count += 1

        preview_rows.append({
            "Row": int(row["row_num"]),
            "Asset": asset_class,
            "Instrument": symbol_or_code,
            "Action": action,
            "Qty": quantity,
            "Price": float(row["price"]),
            "Date": row["transaction_date"],
            "Status": status,
            "Reason": reason
        })

    summary = {
        "import": import_count,
        "skip": skip_count,
        "fail": fail_count,
        "total": len(parsed_df)
    }
    return pd.DataFrame(preview_rows), summary

def apply_import_transaction(import_row, skip_existing_duplicates=True):
    """Apply one canonical imported transaction into session-state portfolios."""
    asset_class = import_row["asset_class"]
    action = import_row["action"]
    quantity = float(import_row["quantity"])
    price = float(import_row["price"])
    transaction_date = import_row["transaction_date"]
    import_key = import_row.get("import_key", "")

    if skip_existing_duplicates and import_key_exists(import_key):
        return False, "Duplicate transaction skipped"

    if asset_class == "US Stock":
        ticker = import_row["symbol"]
        try:
            idx = st.session_state.us_portfolio['Ticker'].index(ticker)
            existing_shares = float(st.session_state.us_portfolio['Shares'][idx])
            existing_avg = float(st.session_state.us_portfolio['Avg_Cost'][idx])

            if action == "Buy":
                total_cost = (existing_shares * existing_avg) + (quantity * price)
                new_shares = existing_shares + quantity
                st.session_state.us_portfolio['Shares'][idx] = new_shares
                st.session_state.us_portfolio['Avg_Cost'][idx] = total_cost / new_shares
                log_transaction(ticker, "Buy", quantity, price, "US Stock", transaction_date=transaction_date, import_key=import_key)
            else:
                if quantity > existing_shares:
                    return False, f"Insufficient shares for {ticker}"
                log_transaction(ticker, "Sell", quantity, price, "US Stock", existing_avg, transaction_date=transaction_date, import_key=import_key)
                new_shares = existing_shares - quantity
                if new_shares == 0:
                    del st.session_state.us_portfolio['Ticker'][idx]
                    del st.session_state.us_portfolio['Shares'][idx]
                    del st.session_state.us_portfolio['Avg_Cost'][idx]
                else:
                    st.session_state.us_portfolio['Shares'][idx] = new_shares
            return True, "ok"
        except ValueError:
            if action == "Buy":
                st.session_state.us_portfolio['Ticker'].append(ticker)
                st.session_state.us_portfolio['Shares'].append(quantity)
                st.session_state.us_portfolio['Avg_Cost'].append(price)
                log_transaction(ticker, "Buy", quantity, price, "US Stock", transaction_date=transaction_date, import_key=import_key)
                return True, "ok"
            return False, f"No existing position for {ticker}"

    if asset_class == "Thai Stock":
        ticker = import_row["symbol"]
        try:
            idx = st.session_state.thai_stocks['Ticker'].index(ticker)
            existing_shares = float(st.session_state.thai_stocks['Shares'][idx])
            existing_avg = float(st.session_state.thai_stocks['Avg_Cost'][idx])

            if action == "Buy":
                total_cost = (existing_shares * existing_avg) + (quantity * price)
                new_shares = existing_shares + quantity
                st.session_state.thai_stocks['Shares'][idx] = new_shares
                st.session_state.thai_stocks['Avg_Cost'][idx] = total_cost / new_shares
                log_transaction(ticker, "Buy", quantity, price, "Thai Stock", transaction_date=transaction_date, import_key=import_key)
            else:
                if quantity > existing_shares:
                    return False, f"Insufficient shares for {ticker}"
                log_transaction(ticker, "Sell", quantity, price, "Thai Stock", existing_avg, transaction_date=transaction_date, import_key=import_key)
                new_shares = existing_shares - quantity
                if new_shares == 0:
                    del st.session_state.thai_stocks['Ticker'][idx]
                    del st.session_state.thai_stocks['Shares'][idx]
                    del st.session_state.thai_stocks['Avg_Cost'][idx]
                else:
                    st.session_state.thai_stocks['Shares'][idx] = new_shares
            return True, "ok"
        except ValueError:
            if action == "Buy":
                st.session_state.thai_stocks['Ticker'].append(ticker)
                st.session_state.thai_stocks['Shares'].append(quantity)
                st.session_state.thai_stocks['Avg_Cost'].append(price)
                log_transaction(ticker, "Buy", quantity, price, "Thai Stock", transaction_date=transaction_date, import_key=import_key)
                return True, "ok"
            return False, f"No existing position for {ticker}"

    # Mutual Fund
    fund_code = import_row["fund_code"]
    master = import_row.get("master", "N/A") or "N/A"
    fund_idx = None
    for index, fund in enumerate(st.session_state.vault_portfolio):
        if fund.get('Code') == fund_code:
            fund_idx = index
            break

    if fund_idx is not None:
        existing_units = float(st.session_state.vault_portfolio[fund_idx]['Units'])
        existing_cost = float(st.session_state.vault_portfolio[fund_idx]['Cost'])
        existing_master = str(st.session_state.vault_portfolio[fund_idx].get('Master', 'N/A') or 'N/A').strip().upper()
        normalized_master = str(master or "N/A").strip().upper()
        if action == "Buy":
            total_cost = (existing_units * existing_cost) + (quantity * price)
            new_units = existing_units + quantity
            st.session_state.vault_portfolio[fund_idx]['Units'] = new_units
            st.session_state.vault_portfolio[fund_idx]['Cost'] = total_cost / new_units
            if existing_master in ["", "N/A"] and normalized_master not in ["", "N/A"]:
                st.session_state.vault_portfolio[fund_idx]['Master'] = normalized_master
            log_transaction(
                fund_code,
                "Buy",
                quantity,
                price,
                "Mutual Fund",
                transaction_date=transaction_date,
                import_key=import_key,
                master=st.session_state.vault_portfolio[fund_idx].get('Master', normalized_master),
            )
            return True, "ok"

        if quantity > existing_units:
            return False, f"Insufficient units for {fund_code}"
        log_transaction(
            fund_code,
            "Sell",
            quantity,
            price,
            "Mutual Fund",
            existing_cost,
            transaction_date=transaction_date,
            import_key=import_key,
            master=existing_master,
        )
        new_units = existing_units - quantity
        if new_units == 0:
            del st.session_state.vault_portfolio[fund_idx]
        else:
            st.session_state.vault_portfolio[fund_idx]['Units'] = new_units
        return True, "ok"

    if action == "Buy":
        st.session_state.vault_portfolio.append({
            "Code": fund_code,
            "Units": quantity,
            "Cost": price,
            "Master": master
        })
        log_transaction(
            fund_code,
            "Buy",
            quantity,
            price,
            "Mutual Fund",
            transaction_date=transaction_date,
            import_key=import_key,
            master=master,
        )
        return True, "ok"

    return False, f"No existing position for {fund_code}"

def reverse_transaction_by_index(txn_index, reason="Manual correction"):
    """Create an opposite transaction to reverse a previous one (append-only audit trail)."""
    if txn_index < 0 or txn_index >= len(st.session_state.transaction_history):
        return False, "Invalid transaction selection"

    original = st.session_state.transaction_history[txn_index]

    if original.get("correction_of") is not None:
        return False, "Selected row is already a correction transaction"

    if original.get("reversed_by") is not None:
        return False, "Selected transaction was already reversed"

    original_type = str(original.get("type", "")).strip().title()
    if original_type not in ["Buy", "Sell"]:
        return False, "Only Buy/Sell transactions can be reversed"

    reverse_action = "Sell" if original_type == "Buy" else "Buy"
    asset_class = original.get("asset_type", "US Stock")
    ticker = original.get("ticker", "")
    shares = float(original.get("shares", 0))
    price = float(original.get("price", 0))
    if shares <= 0 or price <= 0 or not ticker:
        return False, "Original transaction has invalid data"

    reverse_row = {
        "asset_class": asset_class,
        "action": reverse_action,
        "symbol": ticker if asset_class != "Mutual Fund" else "",
        "fund_code": ticker if asset_class == "Mutual Fund" else "",
        "quantity": shares,
        "price": price,
        "transaction_date": datetime.now().strftime("%Y-%m-%d"),
        "master": "N/A"
    }

    ok, message = apply_import_transaction(reverse_row, skip_existing_duplicates=False)
    if not ok:
        return False, message

    reverse_index = len(st.session_state.transaction_history) - 1
    st.session_state.transaction_history[txn_index]["reversed_by"] = reverse_index
    st.session_state.transaction_history[reverse_index]["correction_of"] = txn_index
    st.session_state.transaction_history[reverse_index]["correction_reason"] = reason
    save_transaction_history()
    return True, "ok"

# --- EXECUTION ---
df_us = get_stock_data(us_portfolio)
df_thai = get_stock_data(thai_stocks)
df_vault = get_fund_data(vault_portfolio)

# Precompute alerts once per render (for Today Brief and News tab)
us_prices = df_us['Live Price'].tolist() if len(df_us) > 0 else []
thai_prices = df_thai['Live Price'].tolist() if len(df_thai) > 0 else []
us_alerts = check_price_alerts(us_portfolio, us_prices, "US Stock") if us_prices else []
thai_alerts = check_price_alerts(thai_stocks, thai_prices, "Thai Stock") if thai_prices else []
all_alerts = us_alerts + thai_alerts
new_alert_count = sum(1 for alert in all_alerts if alert.get("is_new"))

# Totals
usd_thb_rate = 34.0
grand_total = (df_us['Value'].sum() * usd_thb_rate) + df_thai['Value'].sum() + df_vault['Value'].sum()
grand_cost = (df_us['Cost Basis'].sum() * usd_thb_rate) + df_thai['Cost Basis'].sum() + df_vault['Cost Basis'].sum()
grand_pl = grand_total - grand_cost
grand_pct = (grand_pl / grand_cost) * 100 if grand_cost != 0 else 0

# Today brief helpers
biggest_mover_label = "N/A"
biggest_mover_pct = 0.0
try:
    mover_candidates = []
    if len(df_us) > 0:
        for _, row in df_us.iterrows():
            mover_candidates.append((row['Ticker'], float(row['P/L %'])))
    if len(df_thai) > 0:
        for _, row in df_thai.iterrows():
            mover_candidates.append((row['Ticker'], float(row['P/L %'])))
    if not mover_candidates and len(df_vault) > 0:
        for _, row in df_vault.iterrows():
            mover_candidates.append((row['Code'], float(row['Fund Day Gain %'])))
    if mover_candidates:
        biggest_mover_label, biggest_mover_pct = max(mover_candidates, key=lambda item: abs(item[1]))
except Exception:
    pass

next_event_text = "No upcoming earnings"
try:
    earnings_map = get_earnings_dates(df_us['Ticker'].tolist() if len(df_us) > 0 else [])
    upcoming = []
    for ticker, event_value in earnings_map.items():
        candidate = event_value[0] if isinstance(event_value, list) and len(event_value) > 0 else event_value
        parsed_event = pd.to_datetime(candidate, errors='coerce')
        if not pd.isna(parsed_event):
            upcoming.append((parsed_event, ticker))
    if upcoming:
        upcoming.sort(key=lambda item: item[0])
        event_date, event_ticker = upcoming[0]
        next_event_text = f"{event_ticker} · {event_date.strftime('%b %d')}"
except Exception:
    pass

# --- DASHBOARD ---
st.subheader("🧭 TODAY BRIEF")
brief1, brief2, brief3, brief4 = st.columns(4, gap="small")

with brief1:
    st.metric("Portfolio Move", f"{grand_pct:+.2f}%")

with brief2:
    st.metric("Biggest Mover", f"{biggest_mover_label}", delta=f"{biggest_mover_pct:+.2f}%")

with brief3:
    st.metric("Next Event", next_event_text)

with brief4:
    st.metric("New Alerts", new_alert_count, delta=f"{len(all_alerts)} active")

st.markdown("---")
st.subheader("📊 PORTFOLIO OVERVIEW")
col1, col2, col3 = st.columns(3, gap="medium")

us_pl_usd = df_us['P/L'].sum()
vault_pl = df_vault['P/L'].sum()

with col1:
    st.metric(
        "🛡️ NET WORTH (THB)",
        f"฿{grand_total:,.0f}",
        delta=f"{grand_pct:+.2f}% (฿{grand_pl:,.0f})",
        delta_color="normal" if grand_pct > 0 else "inverse"
    )

with col2:
    us_pct_mean = df_us['P/L %'].mean() if len(df_us) > 0 else 0.0
    st.metric(
        "🦅 US ATTACK",
        f"${df_us['Value'].sum():,.0f}",
        delta=f"{us_pct_mean:+.2f}% Avg (${us_pl_usd:,.0f})",
        delta_color="normal" if us_pct_mean > 0 else "inverse"
    )

with col3:
    vault_pct_mean = df_vault['P/L %'].mean() if len(df_vault) > 0 else 0.0
    st.metric(
        "🏦 THAI VAULT",
        f"฿{df_vault['Value'].sum():,.0f}",
        delta=f"{vault_pct_mean:+.2f}% Avg (฿{vault_pl:,.0f})",
        delta_color="normal" if vault_pct_mean > 0 else "inverse"
    )

st.markdown("---")

menu_structure = {
    "Dashboard": ["Overview"],
    "Portfolio": ["US Equities", "Thai Portfolio", "Analytics"],
    "Market Intelligence": ["News Watchtower", "Market Tools"],
    "Trading Journal": ["Transaction History", "Trade Entry", "CSV Import", "Calendar"],
}

view_to_main_menu = {
    view_name: main_name
    for main_name, view_names in menu_structure.items()
    for view_name in view_names
}

st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0.5rem; margin-bottom: 0.5rem;">
    <div style="background: #3fb950; color: white; width: 20px; height: 20px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px;">S</div>
    <div style="font-weight: 600; font-size: 0.9rem; color: #EAEAEA;">Sniper OS</div>
</div>
""", unsafe_allow_html=True)

view_icons = {
    "Overview": "📊",
    "US Equities": "🦅",
    "Thai Portfolio": "🏦",
    "Analytics": "📈",
    "News Watchtower": "📰",
    "Market Tools": "🧠",
    "Transaction History": "📚",
    "Trade Entry": "✍️",
    "CSV Import": "📥",
    "Calendar": "🗓️",
}

all_views = [view for view_names in menu_structure.values() for view in view_names]
query_view = st.query_params.get("view", "")
if isinstance(query_view, list):
    query_view = query_view[0] if query_view else ""
if query_view in all_views:
    st.session_state.nav_sub_menu = query_view

if "nav_sub_menu" not in st.session_state or st.session_state.nav_sub_menu not in all_views:
    st.session_state.nav_sub_menu = "Overview"

if query_view != st.session_state.nav_sub_menu:
    st.query_params["view"] = st.session_state.nav_sub_menu

for main_name, view_names in menu_structure.items():
    st.sidebar.markdown(f'<div class="sidebar-nav-section">{main_name}</div>', unsafe_allow_html=True)
    for view_name in view_names:
        is_active = st.session_state.nav_sub_menu == view_name
        icon = view_icons.get(view_name, "•")
        display_label = f"{icon}  {view_name}"
        if is_active:
            st.sidebar.markdown(f'<div class="sidebar-nav-active">{display_label}</div>', unsafe_allow_html=True)
            continue
        if st.sidebar.button(display_label, key=f"nav_btn_{main_name}_{view_name}", use_container_width=True):
            st.session_state.nav_sub_menu = view_name
            st.query_params["view"] = view_name
            st.rerun()

selected_view = st.session_state.nav_sub_menu
main_menu = view_to_main_menu.get(selected_view, "Dashboard")

st.caption(f"📍 {main_menu} · {selected_view}")

if selected_view == "Overview":
    st.info("Use the Navigation menu in the sidebar to switch between modules.")

def color_pl(val): return f'color: {"#3fb950" if val > 0 else "#f85149"}; font-weight: bold;'

def render_styled_table(df, formats=None, pl_columns=None, height=None, na_rep="—"):
    style = df.style
    if pl_columns:
        existing_pl_columns = [col for col in pl_columns if col in df.columns]
        if existing_pl_columns:
            style = style.map(color_pl, subset=existing_pl_columns)
    if formats:
        existing_formats = {col: fmt for col, fmt in formats.items() if col in df.columns}
        if existing_formats:
            style = style.format(existing_formats, na_rep=na_rep)

    dataframe_kwargs = {
        "hide_index": True,
        "width": "stretch",
    }
    if height is not None:
        dataframe_kwargs["height"] = height

    st.dataframe(style, **dataframe_kwargs)

if selected_view == "US Equities":
    st.subheader("🦅 US EQUITIES")
    st.caption("Live prices via yfinance · Sorted by ticker")
    render_styled_table(
        df_us,
        formats={
            "Shares": "{:,.2f}",
            "Avg_Cost": "${:,.2f}",
            "Live Price": "${:,.2f}",
            "Value": "${:,.2f}",
            "Cost Basis": "${:,.2f}",
            "P/L": "${:,.2f}",
            "P/L %": "{:.2f}%"
        },
        pl_columns=["P/L %"]
    )

if selected_view == "Thai Portfolio":
    st.subheader("🏦 MUTUAL FUNDS (SEC Direct Data)")
    st.caption("*Real-time NAV from api.sec.or.th · Sorted by Market Value · Master vs Fund % highlights daily relative performance*")

    if len(df_vault) > 0:
        fund_total = len(df_vault)
        avg_day_gain = float(df_vault['Fund Day Gain %'].mean()) if len(df_vault) > 0 else 0.0
        positive_funds = int((df_vault['Fund Day Gain %'] > 0).sum())

        vf1, vf2, vf3 = st.columns(3)
        with vf1:
            st.metric("Funds", fund_total)
        with vf2:
            st.metric("Avg Fund Day %", f"{avg_day_gain:+.2f}%")
        with vf3:
            st.metric("Positive Today", f"{positive_funds}/{fund_total}")

        df_vault_display = df_vault.copy().sort_values("Value", ascending=False)
        df_vault_display = df_vault_display.rename(columns={
            "Code": "Fund Code",
            "Cost": "Avg Cost",
            "Last Price": "NAV",
            "Fund Day Gain %": "Fund Day %",
            "Master": "Master ETF",
            "Master Day Gain %": "Master Day %",
            "Master vs Fund %": "Fund vs Master %",
            "Cost Basis": "Cost Basis",
            "Value": "Market Value",
            "P/L": "Unrealized P/L",
            "P/L %": "Unrealized P/L %",
        })

        table_height = min(40 + len(df_vault_display) * 35 + 12, 820)

        render_styled_table(
            df_vault_display,
            formats={
                "Units": "{:,.2f}",
                "Avg Cost": "฿{:,.4f}",
                "NAV": "฿{:,.4f}",
                "Fund Day %": "{:+.2f}%",
                "Master Day %": "{:+.2f}%",
                "Fund vs Master %": "{:+.2f}%",
                "Cost Basis": "฿{:,.2f}",
                "Market Value": "฿{:,.2f}",
                "Unrealized P/L": "฿{:,.2f}",
                "Unrealized P/L %": "{:+.2f}%"
            },
            pl_columns=["Unrealized P/L %", "Fund Day %", "Master Day %", "Fund vs Master %"],
            height=table_height
        )
    else:
        st.info("No mutual fund positions yet.")
    
    st.markdown("---")
    st.markdown("#### 🏛️ Thai Equities")
    render_styled_table(
        df_thai,
        formats={
            "Shares": "{:,.2f}",
            "Avg_Cost": "฿{:,.2f}",
            "Live Price": "฿{:,.2f}",
            "Value": "฿{:,.2f}",
            "Cost Basis": "฿{:,.2f}",
            "P/L": "฿{:,.2f}",
            "P/L %": "{:.2f}%"
        },
        pl_columns=["P/L %"]
    )

if selected_view == "Analytics":
    st.subheader("📈 PERFORMANCE ANALYSIS")
    st.caption("Portfolio charts, trade journal analytics, risk panel & rebalancing tools")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**US Equities P/L %**")
        st.bar_chart(df_us.set_index('Ticker')['P/L %'])
    
    with col_b:
        st.markdown("**Mutual Funds P/L %**")
        st.bar_chart(df_vault.set_index('Code')['P/L %'])

    st.markdown("---")
    st.markdown("#### Trade Journal Analytics · Phase 1")

    trade_rows = []
    for txn in st.session_state.transaction_history:
        txn_type = str(txn.get("type", "")).strip().title()
        if txn_type != "Sell":
            continue
        txn_date = pd.to_datetime(str(txn.get("date", "")).split(" ")[0], errors="coerce")
        if pd.isna(txn_date):
            continue
        realized = float(txn.get("realized_pl", 0.0) or 0.0)
        risk_amount = txn.get("risk_amount", None)
        risk_float = None
        try:
            risk_float = float(risk_amount) if risk_amount not in [None, ""] else None
        except Exception:
            risk_float = None

        r_multiple = txn.get("r_multiple", None)
        if r_multiple in [None, ""] and risk_float not in [None, 0, 0.0]:
            try:
                r_multiple = realized / risk_float
            except Exception:
                r_multiple = None

        trade_rows.append({
            "Date": txn_date,
            "Ticker": str(txn.get("ticker", "")).strip().upper(),
            "Asset Type": str(txn.get("asset_type", "")).strip(),
            "Session": str(txn.get("session", "N/A") or "N/A").strip(),
            "Strategy": str(txn.get("strategy", "N/A") or "N/A").strip(),
            "Direction": str(txn.get("direction", "N/A") or "N/A").strip(),
            "Followed Rules": bool(txn.get("followed_rules", False)),
            "Confidence": pd.to_numeric(txn.get("confidence", None), errors="coerce"),
            "Realized P/L": realized,
            "Risk Amount": risk_float,
            "R-Multiple": pd.to_numeric(r_multiple, errors="coerce"),
        })

    if trade_rows:
        df_trade = pd.DataFrame(trade_rows).sort_values("Date")
        total_trades = int(len(df_trade))
        win_trades = int((df_trade["Realized P/L"] > 0).sum())
        win_rate = (win_trades / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profit = float(df_trade[df_trade["Realized P/L"] > 0]["Realized P/L"].sum())
        gross_loss = float(df_trade[df_trade["Realized P/L"] < 0]["Realized P/L"].sum())
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None
        expectancy = float(df_trade["Realized P/L"].mean()) if total_trades > 0 else 0.0

        daily_pl = df_trade.groupby(df_trade["Date"].dt.date, as_index=False)["Realized P/L"].sum()
        profitable_days = int((daily_pl["Realized P/L"] > 0).sum()) if len(daily_pl) > 0 else 0
        consistency_score = ((profitable_days / len(daily_pl)) * 100.0) if len(daily_pl) > 0 else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Closed Trades", total_trades)
        with c2:
            st.metric("Win Rate", f"{win_rate:,.1f}%")
        with c3:
            st.metric("Profit Factor", f"{profit_factor:,.2f}" if profit_factor is not None else "—")
        with c4:
            st.metric("Expectancy/Trade", f"{expectancy:,.2f}")
        with c5:
            st.metric("Consistency", f"{consistency_score:,.1f}%")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Session Performance (P/L)**")
            session_pl = df_trade.groupby("Session", as_index=False)["Realized P/L"].sum().sort_values("Realized P/L", ascending=False)
            st.dataframe(session_pl, hide_index=True, width="stretch")
        with d2:
            st.markdown("**Top Symbols (P/L)**")
            symbol_pl = df_trade.groupby("Ticker", as_index=False)["Realized P/L"].sum().sort_values("Realized P/L", ascending=False).head(15)
            st.dataframe(symbol_pl, hide_index=True, width="stretch")

        st.markdown("**Discipline Impact (Followed Rules vs Not)**")
        discipline_rows = []
        for followed, group in df_trade.groupby("Followed Rules"):
            trade_count = int(len(group))
            discipline_rows.append({
                "Followed Rules": "Yes" if followed else "No",
                "Trades": trade_count,
                "Win Rate %": ((group["Realized P/L"] > 0).sum() / trade_count) * 100.0 if trade_count > 0 else 0.0,
                "Avg P/L": float(group["Realized P/L"].mean()) if trade_count > 0 else 0.0,
            })
        if discipline_rows:
            st.dataframe(pd.DataFrame(discipline_rows), hide_index=True, width="stretch")

        if len(daily_pl) > 1:
            daily_pl = daily_pl.sort_values("Date")
            daily_pl["Cumulative P/L"] = daily_pl["Realized P/L"].cumsum()
            st.markdown("**Cumulative Closed-Trade P/L**")
            st.line_chart(daily_pl.set_index("Date")[["Cumulative P/L"]])
    else:
        st.info("No closed trades yet for Phase 1 journal analytics. Log Sell transactions to populate this section.")

    st.markdown("---")
    st.markdown("#### P/L Attribution")

    attribution_rows = []

    if len(df_us) > 0:
        for _, row in df_us.iterrows():
            attribution_rows.append({
                "Asset Class": "US Stock",
                "Instrument": row["Ticker"],
                "Value (THB)": float(row["Value"]) * usd_thb_rate,
                "P/L (THB)": float(row["P/L"]) * usd_thb_rate,
            })

    if len(df_thai) > 0:
        for _, row in df_thai.iterrows():
            attribution_rows.append({
                "Asset Class": "Thai Stock",
                "Instrument": row["Ticker"],
                "Value (THB)": float(row["Value"]),
                "P/L (THB)": float(row["P/L"]),
            })

    if len(df_vault) > 0:
        for _, row in df_vault.iterrows():
            attribution_rows.append({
                "Asset Class": "Mutual Fund",
                "Instrument": row["Code"],
                "Value (THB)": float(row["Value"]),
                "P/L (THB)": float(row["P/L"]),
            })

    if attribution_rows:
        df_attr = pd.DataFrame(attribution_rows)
        df_attr["Contribution %"] = df_attr["P/L (THB)"] / grand_pl * 100 if grand_pl != 0 else 0.0
        df_attr = df_attr.sort_values("P/L (THB)", ascending=False)

        attr_summary = (
            df_attr.groupby("Asset Class", as_index=False)
            .agg({"Value (THB)": "sum", "P/L (THB)": "sum"})
            .sort_values("P/L (THB)", ascending=False)
        )

        c_attr1, c_attr2 = st.columns(2)
        with c_attr1:
            st.markdown("**By Asset Class**")
            st.dataframe(
                attr_summary.style.format({
                    "Value (THB)": "฿{:,.2f}",
                    "P/L (THB)": "฿{:,.2f}",
                }),
                hide_index=True,
                width='stretch'
            )

        with c_attr2:
            st.markdown("**Top Instrument Contributors**")
            st.dataframe(
                df_attr[["Asset Class", "Instrument", "Value (THB)", "P/L (THB)", "Contribution %"]]
                .head(12)
                .style.format({
                    "Value (THB)": "฿{:,.2f}",
                    "P/L (THB)": "฿{:,.2f}",
                    "Contribution %": "{:+.2f}%",
                }),
                hide_index=True,
                width='stretch'
            )

        st.markdown("---")
        st.markdown("#### Attribution History Drilldown")

        # Auto daily snapshot (once per local day)
        today_key = datetime.now().strftime("%Y-%m-%d")
        has_today_auto = False
        for existing_snapshot in st.session_state.analytics_snapshots:
            if str(existing_snapshot.get("snapshot_kind", "")) != "auto_daily":
                continue
            captured_at = str(existing_snapshot.get("captured_at", ""))
            if captured_at.startswith(today_key):
                has_today_auto = True
                break

        if not has_today_auto:
            auto_asset_values = (
                df_attr.groupby("Asset Class", as_index=False)["Value (THB)"]
                .sum()
                .set_index("Asset Class")["Value (THB)"]
                .to_dict()
            )
            auto_asset_pl = (
                df_attr.groupby("Asset Class", as_index=False)["P/L (THB)"]
                .sum()
                .set_index("Asset Class")["P/L (THB)"]
                .to_dict()
            )

            auto_snapshot = {
                "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "snapshot_kind": "auto_daily",
                "net_worth_thb": float(grand_total),
                "grand_pl_thb": float(grand_pl),
                "asset_values": {
                    "US Stock": float(auto_asset_values.get("US Stock", 0.0)),
                    "Thai Stock": float(auto_asset_values.get("Thai Stock", 0.0)),
                    "Mutual Fund": float(auto_asset_values.get("Mutual Fund", 0.0)),
                },
                "asset_pl": {
                    "US Stock": float(auto_asset_pl.get("US Stock", 0.0)),
                    "Thai Stock": float(auto_asset_pl.get("Thai Stock", 0.0)),
                    "Mutual Fund": float(auto_asset_pl.get("Mutual Fund", 0.0)),
                }
            }
            st.session_state.analytics_snapshots.append(auto_snapshot)
            st.session_state.analytics_snapshots = st.session_state.analytics_snapshots[-500:]
            save_analytics_snapshots()

        if st.button("Capture Analytics Snapshot", key="capture_analytics_snapshot"):
            asset_values = (
                df_attr.groupby("Asset Class", as_index=False)["Value (THB)"]
                .sum()
                .set_index("Asset Class")["Value (THB)"]
                .to_dict()
            )
            asset_pl = (
                df_attr.groupby("Asset Class", as_index=False)["P/L (THB)"]
                .sum()
                .set_index("Asset Class")["P/L (THB)"]
                .to_dict()
            )

            snapshot = {
                "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "snapshot_kind": "manual",
                "net_worth_thb": float(grand_total),
                "grand_pl_thb": float(grand_pl),
                "asset_values": {
                    "US Stock": float(asset_values.get("US Stock", 0.0)),
                    "Thai Stock": float(asset_values.get("Thai Stock", 0.0)),
                    "Mutual Fund": float(asset_values.get("Mutual Fund", 0.0)),
                },
                "asset_pl": {
                    "US Stock": float(asset_pl.get("US Stock", 0.0)),
                    "Thai Stock": float(asset_pl.get("Thai Stock", 0.0)),
                    "Mutual Fund": float(asset_pl.get("Mutual Fund", 0.0)),
                }
            }
            st.session_state.analytics_snapshots.append(snapshot)
            st.session_state.analytics_snapshots = st.session_state.analytics_snapshots[-500:]
            save_analytics_snapshots()
            st.success("Snapshot captured")

        snapshots = st.session_state.analytics_snapshots
        if snapshots:
            snapshot_rows = []
            for s in snapshots:
                captured_at = str(s.get("captured_at", ""))
                snapshot_kind = str(s.get("snapshot_kind", "manual"))
                values = s.get("asset_values", {}) if isinstance(s.get("asset_values"), dict) else {}
                pls = s.get("asset_pl", {}) if isinstance(s.get("asset_pl"), dict) else {}
                snapshot_rows.append({
                    "Captured": captured_at,
                    "Type": snapshot_kind,
                    "Net Worth (THB)": float(s.get("net_worth_thb", 0.0) or 0.0),
                    "Total P/L (THB)": float(s.get("grand_pl_thb", 0.0) or 0.0),
                    "US Value (THB)": float(values.get("US Stock", 0.0) or 0.0),
                    "Thai Value (THB)": float(values.get("Thai Stock", 0.0) or 0.0),
                    "Fund Value (THB)": float(values.get("Mutual Fund", 0.0) or 0.0),
                    "US P/L (THB)": float(pls.get("US Stock", 0.0) or 0.0),
                    "Thai P/L (THB)": float(pls.get("Thai Stock", 0.0) or 0.0),
                    "Fund P/L (THB)": float(pls.get("Mutual Fund", 0.0) or 0.0),
                })

            df_snapshots = pd.DataFrame(snapshot_rows)
            df_snapshots["Captured"] = pd.to_datetime(df_snapshots["Captured"], errors="coerce")
            df_snapshots = df_snapshots.dropna(subset=["Captured"]).sort_values("Captured")

            if len(df_snapshots) > 0:
                series_options = {
                    "Net Worth": "Net Worth (THB)",
                    "Total P/L": "Total P/L (THB)",
                    "US Value": "US Value (THB)",
                    "Thai Value": "Thai Value (THB)",
                    "Fund Value": "Fund Value (THB)",
                    "US P/L": "US P/L (THB)",
                    "Thai P/L": "Thai P/L (THB)",
                    "Fund P/L": "Fund P/L (THB)",
                }
                selected_series_label = st.selectbox(
                    "History series",
                    options=list(series_options.keys()),
                    key="attr_history_series"
                )
                selected_col = series_options[selected_series_label]

                history_chart_df = df_snapshots[["Captured", selected_col]].set_index("Captured")
                st.line_chart(history_chart_df)

                last_n = st.slider("Snapshots to display", min_value=5, max_value=200, value=30, step=5, key="snapshots_last_n")
                st.dataframe(
                    df_snapshots.tail(last_n).style.format({
                        "Net Worth (THB)": "฿{:,.2f}",
                        "Total P/L (THB)": "฿{:,.2f}",
                        "US Value (THB)": "฿{:,.2f}",
                        "Thai Value (THB)": "฿{:,.2f}",
                        "Fund Value (THB)": "฿{:,.2f}",
                        "US P/L (THB)": "฿{:,.2f}",
                        "Thai P/L (THB)": "฿{:,.2f}",
                        "Fund P/L (THB)": "฿{:,.2f}",
                    }),
                    hide_index=True,
                    width='stretch'
                )
                st.caption("Type = auto_daily (captured automatically once per day) or manual (captured via button).")
        else:
            st.info("No snapshots yet. Capture your first analytics snapshot to start drilldown history.")
    else:
        st.info("No positions available yet for attribution analysis.")

    st.markdown("---")
    st.markdown("#### Risk Panel v1")

    risk_positions = []

    if len(df_us) > 0:
        for _, row in df_us.iterrows():
            risk_positions.append({
                "Asset Class": "US Stock",
                "Instrument": row["Ticker"],
                "Value (THB)": float(row["Value"]) * usd_thb_rate,
                "PL%": float(row["P/L %"]),
            })

    if len(df_thai) > 0:
        for _, row in df_thai.iterrows():
            risk_positions.append({
                "Asset Class": "Thai Stock",
                "Instrument": row["Ticker"],
                "Value (THB)": float(row["Value"]),
                "PL%": float(row["P/L %"]),
            })

    if len(df_vault) > 0:
        for _, row in df_vault.iterrows():
            risk_positions.append({
                "Asset Class": "Mutual Fund",
                "Instrument": row["Code"],
                "Value (THB)": float(row["Value"]),
                "PL%": float(row["P/L %"]),
            })

    if risk_positions:
        df_risk = pd.DataFrame(risk_positions)
        total_value_thb = float(df_risk["Value (THB)"].sum())

        if total_value_thb > 0:
            df_risk["Weight %"] = (df_risk["Value (THB)"] / total_value_thb) * 100
            df_risk["Weight"] = df_risk["Weight %"] / 100
        else:
            df_risk["Weight %"] = 0.0
            df_risk["Weight"] = 0.0

        df_risk = df_risk.sort_values("Weight %", ascending=False)

        top_weight = float(df_risk["Weight %"].iloc[0]) if len(df_risk) > 0 else 0.0
        top_instrument = str(df_risk["Instrument"].iloc[0]) if len(df_risk) > 0 else "N/A"
        top3_weight = float(df_risk["Weight %"].head(3).sum()) if len(df_risk) > 0 else 0.0
        hhi = float((df_risk["Weight"] ** 2).sum())
        effective_n = (1 / hhi) if hhi > 0 else 0.0

        weighted_pl_pct = float((df_risk["PL%"] * df_risk["Weight"]).sum())
        pl_pct_std = float(df_risk["PL%"].std()) if len(df_risk) > 1 else 0.0

        # Simple stress assumptions for V1
        stress_us = (df_risk[df_risk["Asset Class"] == "US Stock"]["Value (THB)"].sum()) * 0.10
        stress_thai = (df_risk[df_risk["Asset Class"] == "Thai Stock"]["Value (THB)"].sum()) * 0.08
        stress_fund = (df_risk[df_risk["Asset Class"] == "Mutual Fund"]["Value (THB)"].sum()) * 0.06
        stress_loss = float(stress_us + stress_thai + stress_fund)
        stress_loss_pct = (stress_loss / total_value_thb * 100) if total_value_thb > 0 else 0.0

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Top Position", f"{top_weight:.2f}%", delta=top_instrument)
        with r2:
            st.metric("Top 3 Concentration", f"{top3_weight:.2f}%")
        with r3:
            st.metric("Effective # Holdings", f"{effective_n:.2f}")
        with r4:
            st.metric("Stress Loss (V1)", f"฿{stress_loss:,.0f}", delta=f"-{stress_loss_pct:.2f}%")

        rr1, rr2 = st.columns(2)
        with rr1:
            st.markdown("**Exposure Mix**")
            exposure_mix = (
                df_risk.groupby("Asset Class", as_index=False)["Value (THB)"]
                .sum()
                .sort_values("Value (THB)", ascending=False)
            )
            exposure_mix["Weight %"] = (exposure_mix["Value (THB)"] / total_value_thb) * 100 if total_value_thb > 0 else 0.0
            st.dataframe(
                exposure_mix.style.format({
                    "Value (THB)": "฿{:,.2f}",
                    "Weight %": "{:.2f}%",
                }),
                hide_index=True,
                width='stretch'
            )

        with rr2:
            st.markdown("**Risk Snapshot by Position**")
            st.dataframe(
                df_risk[["Asset Class", "Instrument", "Value (THB)", "Weight %", "PL%"]]
                .head(15)
                .style.format({
                    "Value (THB)": "฿{:,.2f}",
                    "Weight %": "{:.2f}%",
                    "PL%": "{:+.2f}%",
                }),
                hide_index=True,
                width='stretch'
            )

        st.caption(
            f"Portfolio weighted P/L: {weighted_pl_pct:+.2f}% • Cross-position P/L dispersion (std): {pl_pct_std:.2f}%"
        )
        st.caption("Stress Loss (V1) assumptions: US -10%, Thai Stock -8%, Mutual Fund -6%.")

        st.markdown("---")
        st.markdown("#### What-if Rebalancing")

        exposure_map = {
            "US Stock": float(df_risk[df_risk["Asset Class"] == "US Stock"]["Value (THB)"].sum()),
            "Thai Stock": float(df_risk[df_risk["Asset Class"] == "Thai Stock"]["Value (THB)"].sum()),
            "Mutual Fund": float(df_risk[df_risk["Asset Class"] == "Mutual Fund"]["Value (THB)"].sum()),
        }

        if total_value_thb > 0:
            current_us_pct = (exposure_map["US Stock"] / total_value_thb) * 100
            current_thai_pct = (exposure_map["Thai Stock"] / total_value_thb) * 100
            current_fund_pct = (exposure_map["Mutual Fund"] / total_value_thb) * 100

            rb_col1, rb_col2, rb_col3 = st.columns(3)
            with rb_col1:
                target_us = st.slider("Target US Stock %", 0, 100, int(round(current_us_pct)), key="rb_target_us")
            with rb_col2:
                target_thai = st.slider("Target Thai Stock %", 0, 100, int(round(current_thai_pct)), key="rb_target_thai")
            with rb_col3:
                target_fund = st.slider("Target Mutual Fund %", 0, 100, int(round(current_fund_pct)), key="rb_target_fund")

            target_total = target_us + target_thai + target_fund
            if target_total != 100:
                st.warning(f"Target weights must sum to 100%. Current total: {target_total}%")
            else:
                rebalance_rows = []
                for asset_class, target_pct in [
                    ("US Stock", target_us),
                    ("Thai Stock", target_thai),
                    ("Mutual Fund", target_fund),
                ]:
                    current_value = exposure_map[asset_class]
                    target_value = total_value_thb * (target_pct / 100)
                    delta_value = target_value - current_value
                    action = "Buy" if delta_value > 0 else "Sell" if delta_value < 0 else "Hold"
                    rebalance_rows.append({
                        "Asset Class": asset_class,
                        "Current %": (current_value / total_value_thb) * 100,
                        "Target %": target_pct,
                        "Current Value (THB)": current_value,
                        "Target Value (THB)": target_value,
                        "Trade Needed (THB)": delta_value,
                        "Action": action,
                    })

                df_rebalance = pd.DataFrame(rebalance_rows)
                st.dataframe(
                    df_rebalance.style.format({
                        "Current %": "{:.2f}%",
                        "Target %": "{:.2f}%",
                        "Current Value (THB)": "฿{:,.2f}",
                        "Target Value (THB)": "฿{:,.2f}",
                        "Trade Needed (THB)": "฿{:+,.2f}",
                    }),
                    hide_index=True,
                    width='stretch'
                )
                st.caption("Positive trade values indicate buy amount required; negative values indicate sell amount required.")

        st.markdown("---")
        st.markdown("#### Scenario Backtesting")

        scenario_presets = {
            "Soft Landing": {"US Stock": -3.0, "Thai Stock": -2.0, "Mutual Fund": -1.5},
            "US Tech Drawdown": {"US Stock": -15.0, "Thai Stock": -5.0, "Mutual Fund": -8.0},
            "Thailand Risk-off": {"US Stock": -4.0, "Thai Stock": -12.0, "Mutual Fund": -7.0},
            "Global Risk-off": {"US Stock": -18.0, "Thai Stock": -14.0, "Mutual Fund": -10.0},
            "Risk-on Rally": {"US Stock": 10.0, "Thai Stock": 7.0, "Mutual Fund": 6.0},
        }

        selected_preset = st.selectbox(
            "Scenario preset",
            options=list(scenario_presets.keys()),
            key="scenario_backtest_preset"
        )
        preset_shocks = scenario_presets[selected_preset]

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            shock_us = st.number_input("US shock %", value=float(preset_shocks["US Stock"]), step=0.5, key="scenario_shock_us")
        with sc2:
            shock_thai = st.number_input("Thai shock %", value=float(preset_shocks["Thai Stock"]), step=0.5, key="scenario_shock_thai")
        with sc3:
            shock_fund = st.number_input("Fund shock %", value=float(preset_shocks["Mutual Fund"]), step=0.5, key="scenario_shock_fund")

        backtest_rows = []
        projected_total = 0.0
        for asset_class, shock_pct in [
            ("US Stock", shock_us),
            ("Thai Stock", shock_thai),
            ("Mutual Fund", shock_fund),
        ]:
            current_value = exposure_map.get(asset_class, 0.0)
            projected_value = current_value * (1 + (float(shock_pct) / 100.0))
            delta_value = projected_value - current_value
            projected_total += projected_value
            backtest_rows.append({
                "Asset Class": asset_class,
                "Current Value (THB)": current_value,
                "Shock %": float(shock_pct),
                "Projected Value (THB)": projected_value,
                "Delta (THB)": delta_value,
            })

        projected_pl = projected_total - grand_cost
        projected_pct = (projected_pl / grand_cost * 100) if grand_cost != 0 else 0.0

        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Current Net Worth", f"฿{grand_total:,.0f}")
        with b2:
            st.metric("Projected Net Worth", f"฿{projected_total:,.0f}", delta=f"฿{projected_total - grand_total:+,.0f}")
        with b3:
            st.metric("Projected Portfolio P/L", f"฿{projected_pl:,.0f}", delta=f"{projected_pct:+.2f}%")

        st.dataframe(
            pd.DataFrame(backtest_rows).style.format({
                "Current Value (THB)": "฿{:,.2f}",
                "Shock %": "{:+.2f}%",
                "Projected Value (THB)": "฿{:,.2f}",
                "Delta (THB)": "฿{:+,.2f}",
            }),
            hide_index=True,
            width='stretch'
        )

        st.markdown("**Save / Replay Scenarios**")
        scenario_name = st.text_input("Scenario name", value=selected_preset, key="scenario_save_name")
        save_col, replay_col = st.columns(2)

        with save_col:
            if st.button("Save Current Scenario", key="save_current_scenario"):
                clean_name = str(scenario_name).strip()
                if not clean_name:
                    st.warning("Please provide a scenario name.")
                else:
                    new_entry = {
                        "name": clean_name,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "shocks": {
                            "US Stock": float(shock_us),
                            "Thai Stock": float(shock_thai),
                            "Mutual Fund": float(shock_fund),
                        }
                    }
                    without_same = [s for s in st.session_state.saved_scenarios if str(s.get("name", "")) != clean_name]
                    without_same.append(new_entry)
                    st.session_state.saved_scenarios = without_same[-200:]
                    save_saved_scenarios()
                    st.success(f"Saved scenario: {clean_name}")

        saved_names = [str(s.get("name", "Untitled")) for s in st.session_state.saved_scenarios]
        with replay_col:
            if saved_names:
                selected_saved = st.selectbox("Saved scenario", options=saved_names, key="saved_scenario_select")
                if st.button("Replay Selected", key="replay_selected_scenario"):
                    chosen = next((s for s in st.session_state.saved_scenarios if str(s.get("name", "")) == selected_saved), None)
                    if chosen:
                        shocks = chosen.get("shocks", {}) if isinstance(chosen.get("shocks"), dict) else {}
                        st.session_state["scenario_shock_us"] = float(shocks.get("US Stock", shock_us))
                        st.session_state["scenario_shock_thai"] = float(shocks.get("Thai Stock", shock_thai))
                        st.session_state["scenario_shock_fund"] = float(shocks.get("Mutual Fund", shock_fund))
                        st.rerun()
            else:
                st.caption("No saved scenarios yet.")

        st.markdown("**Scenario Compare**")
        compare_candidates = [{
            "name": "Current Inputs",
            "shocks": {
                "US Stock": float(shock_us),
                "Thai Stock": float(shock_thai),
                "Mutual Fund": float(shock_fund),
            }
        }]
        compare_candidates.extend(st.session_state.saved_scenarios)

        compare_options = [str(c.get("name", "Untitled")) for c in compare_candidates]
        default_compare = compare_options[: min(3, len(compare_options))]
        selected_compare = st.multiselect(
            "Select scenarios to compare",
            options=compare_options,
            default=default_compare,
            key="scenario_compare_multiselect"
        )

        compare_rows = []
        for scenario_label in selected_compare:
            scenario = next((c for c in compare_candidates if str(c.get("name", "")) == scenario_label), None)
            if not scenario:
                continue
            shocks = scenario.get("shocks", {}) if isinstance(scenario.get("shocks"), dict) else {}

            scenario_projected_total = 0.0
            for asset_class in ["US Stock", "Thai Stock", "Mutual Fund"]:
                shock_pct = float(shocks.get(asset_class, 0.0) or 0.0)
                current_value = float(exposure_map.get(asset_class, 0.0) or 0.0)
                scenario_projected_total += current_value * (1 + (shock_pct / 100.0))

            scenario_projected_pl = scenario_projected_total - grand_cost
            scenario_projected_pct = (scenario_projected_pl / grand_cost * 100) if grand_cost != 0 else 0.0

            compare_rows.append({
                "Scenario": scenario_label,
                "Projected Net Worth (THB)": scenario_projected_total,
                "Delta vs Current (THB)": scenario_projected_total - grand_total,
                "Projected P/L (THB)": scenario_projected_pl,
                "Projected P/L %": scenario_projected_pct,
            })

        if compare_rows:
            st.dataframe(
                pd.DataFrame(compare_rows)
                .sort_values("Projected Net Worth (THB)", ascending=False)
                .style.format({
                    "Projected Net Worth (THB)": "฿{:,.2f}",
                    "Delta vs Current (THB)": "฿{:+,.2f}",
                    "Projected P/L (THB)": "฿{:,.2f}",
                    "Projected P/L %": "{:+.2f}%",
                }),
                hide_index=True,
                width='stretch'
            )

        st.markdown("---")
        st.markdown("#### Signal Scoring Layer")

        alert_tickers = {str(alert.get("ticker", "")).upper() for alert in all_alerts}
        score_rows = []

        for _, row in df_risk.iterrows():
            pl_pct = float(row.get("PL%", 0.0))
            weight_pct = float(row.get("Weight %", 0.0))
            instrument = str(row.get("Instrument", "")).upper()

            momentum_component = max(-20.0, min(20.0, pl_pct)) * 1.5
            concentration_penalty = max(0.0, weight_pct - 15.0) * 0.8
            alert_adjustment = -6.0 if instrument in alert_tickers else 0.0

            raw_score = 50.0 + momentum_component - concentration_penalty + alert_adjustment
            score = max(0.0, min(100.0, raw_score))

            if score >= 65:
                signal = "Bullish"
            elif score <= 35:
                signal = "Bearish"
            else:
                signal = "Neutral"

            score_rows.append({
                "Asset Class": row["Asset Class"],
                "Instrument": row["Instrument"],
                "Weight %": weight_pct,
                "P/L %": pl_pct,
                "Signal Score": score,
                "Signal": signal,
            })

        if score_rows:
            df_scores = pd.DataFrame(score_rows).sort_values("Signal Score", ascending=False)
            st.dataframe(
                df_scores.style.format({
                    "Weight %": "{:.2f}%",
                    "P/L %": "{:+.2f}%",
                    "Signal Score": "{:.1f}",
                }),
                hide_index=True,
                width='stretch'
            )
            st.caption("Scoring inputs: capped P/L momentum, concentration penalty above 15% position weight, and active alert penalty.")
    else:
        st.info("No positions available yet for risk analysis.")

if selected_view == "News Watchtower":
    st.subheader("📰 NEWS WATCHTOWER")
    st.caption("Real-time news and alerts for your holdings")
    
    # Get all tickers from portfolios
    all_tickers = us_portfolio.get("Ticker", []) + thai_stocks.get("Ticker", [])
    
    if not all_tickers:
        st.info("📌 Add holdings to see news and alerts for your positions")
    else:
        # Check if NewsAPI key is configured
        newsapi_key = get_newsapi_key()
        
        if not newsapi_key:
            st.warning(
                "⚠️ NewsAPI key not configured. Add your free key from https://newsapi.org/ "
                "in `.streamlit/secrets.toml` (`[news_alerts] -> newsapi_key`) "
                "or set environment variable `NEWSAPI_KEY`."
            )
        else:
            # Price Alerts Section
            st.markdown("#### 🚨 PRICE ALERTS")
            
            if all_alerts:
                for alert in all_alerts:
                    col_alert1, col_alert2, col_alert3 = st.columns([1, 2, 2])
                    with col_alert1:
                        st.metric(
                            f"{alert['ticker']}{' 🆕' if alert.get('is_new') else ''}",
                            f"{alert['change_pct']:+.2f}%"
                        )
                    with col_alert2:
                        st.write(f"**Current:** {alert['currency_symbol']}{alert['current_price']:.2f}")
                    with col_alert3:
                        st.write(f"**Cost Base:** {alert['currency_symbol']}{alert['price_at_cost']:.2f}")
            else:
                st.success("✅ No price alerts - all holdings within threshold")
            
            st.markdown("---")
            
            # News Feed Section
            st.markdown("#### 📺 NEWS FEED")
            selected_ticker = st.selectbox("Select holding to view news", all_tickers)
            
            if selected_ticker:
                st.write(f"**Latest news for {selected_ticker}**")
                news_articles = fetch_news_for_ticker(selected_ticker)
                
                if news_articles:
                    summary_rows = []
                    for article in news_articles:
                        relative_time, exact_time = format_news_timestamp(article.get('publishedAt'))
                        summary_rows.append({
                            "When": relative_time,
                            "Published (Local)": exact_time,
                            "Source": article.get('source', {}).get('name', 'Unknown'),
                            "Headline": article.get('title', 'Untitled')
                        })

                    st.markdown("**At a glance**")
                    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

                    st.markdown("**Details**")
                    for i, article in enumerate(news_articles):
                        relative_time, exact_time = format_news_timestamp(article.get('publishedAt'))
                        source_name = article.get('source', {}).get('name', 'Unknown')
                        title = article.get('title', 'Untitled')
                        with st.expander(f"📰 [{relative_time}] {title} ({source_name})"):
                            st.caption(f"Published: {exact_time}")
                            st.write(article.get('description') or "No description available.")
                            article_url = article.get('url', '')
                            if article_url and str(article_url).startswith(('http://', 'https://')):
                                st.markdown(f"[Read Full Article]({article_url})")
                            else:
                                st.caption("Article link unavailable")
                else:
                    st.info(f"No recent news found for {selected_ticker}")

if selected_view == "Transaction History":
    st.subheader("📜 TRANSACTION HISTORY & REALIZED P/L")
    
    # Summary metrics
    realized_usd, realized_thb = get_realized_pl_summary()
    
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    
    with col_summary1:
        st.metric("💵 Realized P/L (USD)", f"${realized_usd:,.2f}")
    
    with col_summary2:
        st.metric("💶 Realized P/L (THB)", f"฿{realized_thb:,.2f}")
    
    with col_summary3:
        total_transactions = len(st.session_state.transaction_history)
        st.metric("📊 Total Transactions", total_transactions)
    
    st.markdown("---")
    
    # Transaction history table
    if st.session_state.transaction_history:
        st.markdown("#### All Transactions")

        reversible_options = []
        for idx, txn in enumerate(st.session_state.transaction_history):
            if txn.get("correction_of") is not None:
                continue
            if txn.get("reversed_by") is not None:
                continue
            txn_type = str(txn.get("type", "")).strip().title()
            if txn_type not in ["Buy", "Sell"]:
                continue
            reversible_options.append((idx, txn))

        if reversible_options:
            st.markdown("#### Transaction Corrections")
            option_map = {}
            for idx, txn in reversed(reversible_options[-200:]):
                label = (
                    f"#{idx + 1} · {txn.get('date', 'N/A')} · "
                    f"{txn.get('ticker', 'N/A')} · {txn.get('type', 'N/A')} "
                    f"{float(txn.get('shares', 0)):,.4f} @ {float(txn.get('price', 0)):,.4f}"
                )
                option_map[label] = idx

            correction_col1, correction_col2 = st.columns([3, 2])
            with correction_col1:
                selected_label = st.selectbox(
                    "Select transaction to reverse",
                    options=list(option_map.keys()),
                    key="txn_reverse_select"
                )
            with correction_col2:
                correction_reason = st.text_input(
                    "Reason",
                    value="Manual correction",
                    key="txn_reverse_reason"
                )

            if st.button("Reverse Selected Transaction", key="txn_reverse_button"):
                target_index = option_map[selected_label]
                ok, message = reverse_transaction_by_index(target_index, correction_reason)
                if ok:
                    st.success("Transaction reversed and correction logged")
                    st.rerun()
                else:
                    st.error(f"Unable to reverse transaction: {message}")
        else:
            st.caption("No reversible transactions available.")
        
        df_txn = get_transaction_dataframe()
        
        # Sort by date descending (most recent first)
        df_txn = df_txn.sort_values("Date", ascending=False)
        
        # Display with formatting
        df_txn_display = df_txn.copy()
        df_txn_display["Price Display"] = df_txn_display.apply(
            lambda row: f"${row['Price']:,.2f}" if row["Currency"] == "USD" else f"฿{row['Price']:,.2f}",
            axis=1
        )
        df_txn_display["Total Display"] = df_txn_display.apply(
            lambda row: f"${row['Total']:,.2f}" if row["Currency"] == "USD" else f"฿{row['Total']:,.2f}",
            axis=1
        )
        df_txn_display["Realized P/L Display"] = df_txn_display.apply(
            lambda row: f"${row['Realized P/L']:,.2f}" if row["Currency"] == "USD" else f"฿{row['Realized P/L']:,.2f}",
            axis=1
        )

        st.dataframe(
            df_txn_display[[
                "Date", "Ticker", "Type", "Shares", "Price Display", "Total Display",
                "Asset Type", "Currency", "Lot Method", "Realized P/L Display",
                "Strategy", "Session", "Direction", "Followed Rules", "Confidence", "R-Multiple"
            ]],
            hide_index=True,
            width="stretch"
        )
        
        # Export option
        csv = df_txn.to_csv(index=False)
        st.download_button(
            label="📥 Download Transaction History (CSV)",
            data=csv,
            file_name="sniper_transactions.csv",
            mime="text/csv"
        )
    else:
        st.info("📌 No transactions yet. Use the Trade Entry tab to log buy/sell activity and track realized P/L.")
    
    # Explanation
    with st.expander("ℹ️ Understanding Realized vs Unrealized P/L"):
        st.markdown("""
        **Unrealized P/L (Shown in portfolio tabs):**
        - Paper gains/losses on positions you still hold
        - Calculated as: (Current Price - Average Cost) × Shares
        - Changes with market prices
        
        **Realized P/L (Shown here):**
        - Actual gains/losses from completed sales
        - Calculated as: (Sale Price - Average Cost at Time) × Shares Sold
        - Locked in and doesn't change
        - Used for tax reporting
        
        **Note:** Current implementation uses weighted average cost basis. Future updates will support FIFO/LIFO tax lot methods.
        """)

if selected_view == "Market Tools":
    st.subheader("🔎 MARKET TOOLS")
    st.caption("Phase 1 live: Watchlists, Compare, Screener · Phase 2/3 scaffolded: Fundamentals and Options workflows")

    mt1, mt2, mt3, mt4, mt5 = st.tabs([
        "📋 WATCHLISTS",
        "⚖️ COMPARE",
        "🧪 SCREENER",
        "📈 FUNDAMENTALS (PHASE 2)",
        "📉 OPTIONS (PHASE 3)",
    ])

    with mt1:
        st.markdown("#### Manage Watchlists")
        watchlists = st.session_state.watchlists
        list_names = sorted(list(watchlists.keys()))

        create_col, select_col = st.columns([2, 3])
        with create_col:
            new_list_name = st.text_input("New watchlist name", key="wl_new_name")
            if st.button("Create Watchlist", key="wl_create_btn"):
                clean_name = str(new_list_name).strip()
                if not clean_name:
                    st.error("Enter a watchlist name.")
                elif clean_name in watchlists:
                    st.warning("Watchlist already exists.")
                else:
                    watchlists[clean_name] = []
                    save_watchlists()
                    st.success(f"Created watchlist: {clean_name}")
                    st.rerun()

        selected_watchlist = None
        with select_col:
            if list_names:
                selected_watchlist = st.selectbox("Select watchlist", list_names, key="wl_select")
            else:
                st.info("Create your first watchlist to start tracking symbols.")

        if selected_watchlist:
            symbols = watchlists.get(selected_watchlist, [])
            add_col, remove_col = st.columns([2, 3])

            with add_col:
                symbol_to_add = st.text_input("Add symbol", key="wl_add_symbol")
                if st.button("Add Symbol", key="wl_add_btn"):
                    clean_symbol = _normalize_market_symbol(symbol_to_add)
                    if not clean_symbol:
                        st.error("Enter a valid symbol.")
                    elif clean_symbol in symbols:
                        st.warning("Symbol already in watchlist.")
                    else:
                        watchlists[selected_watchlist] = symbols + [clean_symbol]
                        save_watchlists()
                        st.success(f"Added {clean_symbol} to {selected_watchlist}")
                        st.rerun()

            with remove_col:
                symbols_to_remove = st.multiselect(
                    "Remove symbols",
                    options=symbols,
                    key="wl_remove_symbols"
                )
                if st.button("Remove Selected", key="wl_remove_btn"):
                    keep = [s for s in symbols if s not in symbols_to_remove]
                    watchlists[selected_watchlist] = keep
                    save_watchlists()
                    st.success("Watchlist updated")
                    st.rerun()

            if symbols:
                st.markdown("#### Watchlist Snapshot")
                df_watch = fetch_quote_snapshot(symbols)
                if len(df_watch) > 0:
                    render_styled_table(
                        df_watch,
                        formats={
                            "Price": "{:,.2f}",
                            "Change %": "{:+.2f}%",
                            "Market Cap": "{:,.0f}",
                            "P/E": "{:,.2f}",
                            "Forward P/E": "{:,.2f}",
                            "Div Yield %": "{:,.2f}%",
                            "Beta": "{:,.2f}",
                            "52W High": "{:,.2f}",
                            "52W Low": "{:,.2f}",
                            "Avg Vol": "{:,.0f}",
                        },
                        pl_columns=["Change %"]
                    )
            else:
                st.info("No symbols yet in this watchlist.")

    with mt2:
        st.markdown("#### Multi-Symbol Compare")
        base_universe = _get_base_symbol_universe()
        selected_compare = st.multiselect(
            "Select symbols",
            options=base_universe,
            default=base_universe[: min(6, len(base_universe))],
            key="compare_symbols_select"
        )

        if selected_compare:
            df_compare = fetch_quote_snapshot(selected_compare)
            compare_cols = [
                c for c in [
                    "Symbol", "Price", "Change %", "Market Cap", "P/E", "Forward P/E",
                    "Div Yield %", "Beta", "52W High", "52W Low", "Avg Vol", "Sector"
                ] if c in df_compare.columns
            ]
            if len(df_compare) > 0 and compare_cols:
                render_styled_table(
                    df_compare[compare_cols],
                    formats={
                        "Price": "{:,.2f}",
                        "Change %": "{:+.2f}%",
                        "Market Cap": "{:,.0f}",
                        "P/E": "{:,.2f}",
                        "Forward P/E": "{:,.2f}",
                        "Div Yield %": "{:,.2f}%",
                        "Beta": "{:,.2f}",
                        "52W High": "{:,.2f}",
                        "52W Low": "{:,.2f}",
                        "Avg Vol": "{:,.0f}",
                    },
                    pl_columns=["Change %"]
                )
        else:
            st.info("Select at least one symbol to compare.")

    with mt3:
        st.markdown("#### Rule-Based Screener")
        base_universe = _get_base_symbol_universe()
        default_universe_text = ", ".join(base_universe[: min(40, len(base_universe))])
        universe_text = st.text_area(
            "Universe symbols (comma-separated)",
            value=default_universe_text,
            key="screener_universe_text"
        )

        f1, f2, f3 = st.columns(3)
        with f1:
            min_market_cap_bn = st.number_input("Min Market Cap (USD bn)", min_value=0.0, value=0.0, step=1.0, key="scr_min_mcap")
            max_pe = st.number_input("Max P/E", min_value=0.0, value=1000.0, step=1.0, key="scr_max_pe")
        with f2:
            min_div_yield = st.number_input("Min Dividend Yield %", min_value=0.0, value=0.0, step=0.1, key="scr_min_div")
            max_beta = st.number_input("Max Beta", min_value=0.0, value=10.0, step=0.1, key="scr_max_beta")
        with f3:
            min_change_pct = st.number_input("Min Daily Change %", value=-100.0, step=0.5, key="scr_min_change")
            max_change_pct = st.number_input("Max Daily Change %", value=100.0, step=0.5, key="scr_max_change")

        raw_symbols = [s.strip().upper() for s in universe_text.split(",") if s.strip()]
        screened_symbols = sorted(list(dict.fromkeys(raw_symbols)))

        if st.button("Run Screener", key="run_screener_btn"):
            if not screened_symbols:
                st.error("Provide at least one symbol in the screener universe.")
            else:
                df_screen = fetch_quote_snapshot(screened_symbols)
                if len(df_screen) == 0:
                    st.warning("No quote data available for provided symbols.")
                else:
                    working = df_screen.copy()
                    if "Market Cap" in working.columns:
                        working = working[(working["Market Cap"].isna()) | (working["Market Cap"] >= min_market_cap_bn * 1_000_000_000)]
                    if "P/E" in working.columns:
                        working = working[(working["P/E"].isna()) | (working["P/E"] <= max_pe)]
                    if "Div Yield %" in working.columns:
                        working = working[(working["Div Yield %"].isna()) | (working["Div Yield %"] >= min_div_yield)]
                    if "Beta" in working.columns:
                        working = working[(working["Beta"].isna()) | (working["Beta"] <= max_beta)]
                    if "Change %" in working.columns:
                        working = working[(working["Change %"].isna()) | ((working["Change %"] >= min_change_pct) & (working["Change %"] <= max_change_pct))]

                    st.success(f"Matched {len(working)} / {len(df_screen)} symbols")
                    if len(working) > 0:
                        render_styled_table(
                            working,
                            formats={
                                "Price": "{:,.2f}",
                                "Change %": "{:+.2f}%",
                                "Market Cap": "{:,.0f}",
                                "P/E": "{:,.2f}",
                                "Forward P/E": "{:,.2f}",
                                "Div Yield %": "{:,.2f}%",
                                "Beta": "{:,.2f}",
                                "52W High": "{:,.2f}",
                                "52W Low": "{:,.2f}",
                                "Avg Vol": "{:,.0f}",
                            },
                            pl_columns=["Change %"]
                        )

    with mt4:
        st.markdown("#### Fundamentals Snapshot")
        st.caption("Phase 2 live: full statements, analyst context, earnings surprise tracking, and factor scorecard.")

        fundamentals_universe = _get_base_symbol_universe()
        selected_fund_symbol = st.selectbox(
            "Symbol",
            options=fundamentals_universe if fundamentals_universe else ["AAPL"],
            key="fundamentals_symbol_select"
        )

        fundamentals_package = fetch_fundamental_snapshot(selected_fund_symbol)
        metrics = fundamentals_package.get("metrics", {}) if isinstance(fundamentals_package, dict) else {}
        trend_df = fundamentals_package.get("trend_df", pd.DataFrame()) if isinstance(fundamentals_package, dict) else pd.DataFrame()
        income_df = fundamentals_package.get("income_df", pd.DataFrame()) if isinstance(fundamentals_package, dict) else pd.DataFrame()
        balance_df = fundamentals_package.get("balance_df", pd.DataFrame()) if isinstance(fundamentals_package, dict) else pd.DataFrame()
        cashflow_df = fundamentals_package.get("cashflow_df", pd.DataFrame()) if isinstance(fundamentals_package, dict) else pd.DataFrame()
        recommendations_df = fundamentals_package.get("recommendations_df", pd.DataFrame()) if isinstance(fundamentals_package, dict) else pd.DataFrame()
        earnings_dates_df = fundamentals_package.get("earnings_dates_df", pd.DataFrame()) if isinstance(fundamentals_package, dict) else pd.DataFrame()
        analyst_snapshot = fundamentals_package.get("analyst_snapshot", {}) if isinstance(fundamentals_package, dict) else {}
        factor_scores = fundamentals_package.get("factor_scores", {}) if isinstance(fundamentals_package, dict) else {}

        if metrics:
            metric_items = list(metrics.items())
            m_cols = st.columns(4)
            for idx, (label, value) in enumerate(metric_items):
                with m_cols[idx % 4]:
                    if value is None:
                        st.metric(label, "—")
                    elif "%" in label:
                        st.metric(label, f"{value:,.2f}%")
                    elif label in ["Revenue", "Net Income"]:
                        st.metric(label, f"{value:,.0f}")
                    else:
                        st.metric(label, f"{value:,.2f}")
        else:
            st.info("Fundamental metrics unavailable for this symbol.")

        st.markdown("---")
        st.markdown("**Analyst Context**")
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            st.metric("Recommendation", analyst_snapshot.get("Recommendation") or "—")
        with ac2:
            target_mean = analyst_snapshot.get("Target Mean", None)
            st.metric("Target Mean", f"{target_mean:,.2f}" if target_mean is not None else "—")
        with ac3:
            target_low = analyst_snapshot.get("Target Low", None)
            target_high = analyst_snapshot.get("Target High", None)
            if target_low is not None and target_high is not None:
                st.metric("Target Range", f"{target_low:,.2f} - {target_high:,.2f}")
            else:
                st.metric("Target Range", "—")
        with ac4:
            analyst_count = analyst_snapshot.get("Analyst Opinions", None)
            st.metric("Analyst Opinions", f"{int(analyst_count)}" if analyst_count is not None else "—")

        if isinstance(recommendations_df, pd.DataFrame) and len(recommendations_df) > 0:
            rec_cols = [c for c in ["Date", "Firm", "To Grade", "From Grade", "Action"] if c in recommendations_df.columns]
            if rec_cols:
                st.markdown("**Latest Recommendation Changes**")
                rec_view = recommendations_df.copy()
                if "Date" in rec_view.columns:
                    rec_view["Date"] = pd.to_datetime(rec_view["Date"], errors="coerce")
                    rec_view = rec_view.sort_values("Date", ascending=False)
                st.dataframe(rec_view[rec_cols].head(20), hide_index=True, width="stretch")

        st.markdown("---")
        if len(trend_df) > 0:
            st.markdown("**Income Trend**")
            render_styled_table(
                trend_df,
                formats={
                    "Total Revenue": "{:,.0f}",
                    "Net Income": "{:,.0f}",
                }
            )

        st.markdown("**Financial Statements**")
        statement_view = st.selectbox(
            "Statement",
            ["Income Statement", "Balance Sheet", "Cash Flow"],
            key="fund_statement_view"
        )

        statement_map = {
            "Income Statement": income_df,
            "Balance Sheet": balance_df,
            "Cash Flow": cashflow_df,
        }
        selected_statement_df = statement_map.get(statement_view, pd.DataFrame())

        if isinstance(selected_statement_df, pd.DataFrame) and len(selected_statement_df) > 0:
            preferred_rows = {
                "Income Statement": [
                    "Period", "Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Diluted EPS", "EBITDA"
                ],
                "Balance Sheet": [
                    "Period", "Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity", "Cash And Cash Equivalents", "Current Assets", "Current Liabilities"
                ],
                "Cash Flow": [
                    "Period", "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Free Cash Flow", "Capital Expenditure"
                ],
            }
            keep_cols = [c for c in preferred_rows.get(statement_view, []) if c in selected_statement_df.columns]
            if len(keep_cols) < 2:
                keep_cols = selected_statement_df.columns[: min(8, len(selected_statement_df.columns))].tolist()

            render_styled_table(
                selected_statement_df[keep_cols].head(12),
                formats={
                    col: "{:,.0f}" for col in keep_cols if col != "Period"
                }
            )
        else:
            st.info("Statement data unavailable for this symbol.")

        st.markdown("---")
        st.markdown("**Earnings Surprise & Revisions**")
        if isinstance(earnings_dates_df, pd.DataFrame) and len(earnings_dates_df) > 0:
            earn_view = earnings_dates_df.copy()
            if "Date" in earn_view.columns:
                earn_view["Date"] = pd.to_datetime(earn_view["Date"], errors="coerce")
                earn_view = earn_view.sort_values("Date", ascending=False)

            keep_cols = [c for c in ["Date", "EPS Estimate", "Reported EPS", "Surprise(%)"] if c in earn_view.columns]
            if keep_cols:
                st.dataframe(earn_view[keep_cols].head(8), hide_index=True, width="stretch")

            if "Surprise(%)" in earn_view.columns:
                surprise_series = pd.to_numeric(earn_view["Surprise(%)"], errors="coerce").dropna().head(8)
                if len(surprise_series) > 0:
                    st.bar_chart(pd.DataFrame({"Surprise %": surprise_series.values}))
        else:
            st.info("No earnings surprise history available.")

        st.markdown("---")
        st.markdown("**Factor Scorecard**")
        factor_rows = []
        for factor_name in ["Quality", "Growth", "Value", "Momentum"]:
            score = factor_scores.get(factor_name, None)
            bucket = "N/A"
            if score is not None:
                if score >= 70:
                    bucket = "Strong"
                elif score >= 45:
                    bucket = "Neutral"
                else:
                    bucket = "Weak"
            factor_rows.append({
                "Factor": factor_name,
                "Score": score,
                "Signal": bucket,
            })

        df_factors = pd.DataFrame(factor_rows)
        render_styled_table(
            df_factors,
            formats={"Score": "{:,.1f}"},
            pl_columns=[]
        )

    with mt5:
        st.markdown("#### Options Chain Snapshot")
        st.caption("Phase 3 live: IV analytics, strategy templates, Greeks monitor, and event-driven options alerts.")

        options_universe = _get_base_symbol_universe()
        selected_opt_symbol = st.selectbox(
            "Underlying symbol",
            options=options_universe if options_universe else ["AAPL"],
            key="options_symbol_select"
        )

        expiries, _, _ = fetch_options_snapshot(selected_opt_symbol)
        if expiries:
            selected_expiry = st.selectbox("Expiration", options=expiries, key="options_expiry_select")
            _, calls_df, puts_df = fetch_options_snapshot(selected_opt_symbol, selected_expiry)

            quote_df = fetch_quote_snapshot([selected_opt_symbol])
            underlying_price = None
            if len(quote_df) > 0 and "Price" in quote_df.columns:
                try:
                    underlying_price = float(quote_df.iloc[0]["Price"])
                except Exception:
                    underlying_price = None

            atm_iv = _estimate_atm_iv(calls_df, puts_df, underlying_price)
            _record_atm_iv(selected_opt_symbol, atm_iv)
            iv_rank, iv_percentile, iv_sample = _compute_iv_rank_percentile(selected_opt_symbol, atm_iv)

            ivc1, ivc2, ivc3, ivc4 = st.columns(4)
            with ivc1:
                st.metric("Underlying", f"{underlying_price:,.2f}" if underlying_price is not None else "—")
            with ivc2:
                st.metric("ATM IV", f"{atm_iv * 100:,.2f}%" if atm_iv is not None else "—")
            with ivc3:
                st.metric("IV Rank", f"{iv_rank:,.1f}" if iv_rank is not None else "—")
            with ivc4:
                st.metric("IV Percentile", f"{iv_percentile:,.1f}" if iv_percentile is not None else "—", delta=f"{iv_sample} obs")

            if st.session_state.options_iv_history.get(selected_opt_symbol):
                iv_hist_df = pd.DataFrame(st.session_state.options_iv_history.get(selected_opt_symbol, []))
                if len(iv_hist_df) > 1:
                    iv_hist_df["date"] = pd.to_datetime(iv_hist_df["date"], errors="coerce")
                    iv_hist_df = iv_hist_df.dropna(subset=["date"]).sort_values("date")
                    if len(iv_hist_df) > 1:
                        st.line_chart(iv_hist_df.set_index("date")[["atm_iv"]] * 100.0)

            st.markdown("---")
            st.markdown("**Strategy Templates**")
            strategy_choice = st.selectbox(
                "Template",
                ["Covered Call", "Cash-Secured Put", "Bull Call Spread"],
                key="opt_strategy_template"
            )

            if strategy_choice == "Covered Call":
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    shares_owned = st.number_input("Shares owned", min_value=100, value=100, step=100, key="cc_shares")
                with s2:
                    cc_cost_basis = st.number_input("Cost basis per share", min_value=0.0, value=float(underlying_price or 0.0), step=0.01, key="cc_cost")
                with s3:
                    cc_strike = st.number_input("Short call strike", min_value=0.0, value=float(underlying_price or 0.0), step=0.5, key="cc_strike")
                with s4:
                    cc_premium = st.number_input("Premium received/share", min_value=0.0, value=1.0, step=0.01, key="cc_premium")

                contracts = shares_owned // 100
                total_premium = contracts * 100 * cc_premium
                max_profit = contracts * 100 * max((cc_strike - cc_cost_basis), 0) + total_premium
                breakeven = cc_cost_basis - cc_premium
                st.caption(f"Contracts: {contracts}")
                st.metric("Total Premium", f"{total_premium:,.2f}")
                st.metric("Max Profit", f"{max_profit:,.2f}")
                st.metric("Breakeven", f"{breakeven:,.2f}")

            elif strategy_choice == "Cash-Secured Put":
                s1, s2, s3 = st.columns(3)
                with s1:
                    csp_contracts = st.number_input("Contracts", min_value=1, value=1, step=1, key="csp_contracts")
                with s2:
                    csp_strike = st.number_input("Put strike", min_value=0.0, value=float(underlying_price or 0.0), step=0.5, key="csp_strike")
                with s3:
                    csp_premium = st.number_input("Premium received/share", min_value=0.0, value=1.0, step=0.01, key="csp_premium")

                cash_required = csp_contracts * 100 * csp_strike
                max_profit = csp_contracts * 100 * csp_premium
                breakeven = csp_strike - csp_premium
                st.metric("Cash Required", f"{cash_required:,.2f}")
                st.metric("Max Profit", f"{max_profit:,.2f}")
                st.metric("Breakeven", f"{breakeven:,.2f}")

            else:
                s1, s2, s3, s4, s5 = st.columns(5)
                with s1:
                    spread_contracts = st.number_input("Contracts", min_value=1, value=1, step=1, key="bcs_contracts")
                with s2:
                    long_strike = st.number_input("Long strike", min_value=0.0, value=max(float(underlying_price or 0.0) - 5.0, 0.0), step=0.5, key="bcs_long_strike")
                with s3:
                    long_premium = st.number_input("Long premium/share", min_value=0.0, value=2.0, step=0.01, key="bcs_long_premium")
                with s4:
                    short_strike = st.number_input("Short strike", min_value=0.0, value=float(underlying_price or 0.0) + 5.0, step=0.5, key="bcs_short_strike")
                with s5:
                    short_premium = st.number_input("Short premium/share", min_value=0.0, value=0.8, step=0.01, key="bcs_short_premium")

                net_debit = max((long_premium - short_premium), 0.0)
                width = max(short_strike - long_strike, 0.0)
                max_loss = spread_contracts * 100 * net_debit
                max_profit = spread_contracts * 100 * max(width - net_debit, 0.0)
                breakeven = long_strike + net_debit
                st.metric("Net Debit", f"{net_debit:,.2f}/share")
                st.metric("Max Profit", f"{max_profit:,.2f}")
                st.metric("Max Loss", f"{max_loss:,.2f}")
                st.metric("Breakeven", f"{breakeven:,.2f}")

            oc1, oc2 = st.columns(2)
            with oc1:
                st.markdown("**Top Calls by Open Interest**")
                if isinstance(calls_df, pd.DataFrame) and len(calls_df) > 0:
                    calls_greeks = _annotate_black_scholes_greeks(calls_df, "call", underlying_price, selected_expiry)
                    calls_view = calls_greeks.sort_values("openInterest", ascending=False).head(20)
                    keep_cols = [c for c in ["contractSymbol", "strike", "lastPrice", "impliedVolatility", "Delta", "Gamma", "Theta/day", "Vega", "openInterest", "volume"] if c in calls_view.columns]
                    st.dataframe(calls_view[keep_cols], hide_index=True, width="stretch")
                else:
                    st.info("No calls available")

            with oc2:
                st.markdown("**Top Puts by Open Interest**")
                if isinstance(puts_df, pd.DataFrame) and len(puts_df) > 0:
                    puts_greeks = _annotate_black_scholes_greeks(puts_df, "put", underlying_price, selected_expiry)
                    puts_view = puts_greeks.sort_values("openInterest", ascending=False).head(20)
                    keep_cols = [c for c in ["contractSymbol", "strike", "lastPrice", "impliedVolatility", "Delta", "Gamma", "Theta/day", "Vega", "openInterest", "volume"] if c in puts_view.columns]
                    st.dataframe(puts_view[keep_cols], hide_index=True, width="stretch")
                else:
                    st.info("No puts available")

            st.markdown("---")
            st.markdown("**Event-Driven Options Alerts**")
            alert_col1, alert_col2, alert_col3 = st.columns(3)
            with alert_col1:
                iv_percentile_threshold = st.slider("IV percentile alert ≥", min_value=50, max_value=99, value=80, step=1, key="opt_iv_pct_threshold")
            with alert_col2:
                vol_oi_ratio_threshold = st.slider("Volume/OI ratio alert ≥", min_value=0.5, max_value=5.0, value=1.5, step=0.1, key="opt_vol_oi_threshold")
            with alert_col3:
                earnings_days_window = st.slider("Earnings window (days)", min_value=1, max_value=45, value=14, step=1, key="opt_earnings_window")

            options_alerts = []
            if iv_percentile is not None and iv_percentile >= iv_percentile_threshold:
                options_alerts.append(f"IV percentile is elevated at {iv_percentile:.1f} (threshold {iv_percentile_threshold}).")

            for chain_name, chain_df in [("Calls", calls_df), ("Puts", puts_df)]:
                if not isinstance(chain_df, pd.DataFrame) or len(chain_df) == 0:
                    continue
                if "volume" not in chain_df.columns or "openInterest" not in chain_df.columns:
                    continue
                work = chain_df[["contractSymbol", "volume", "openInterest"]].copy()
                work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
                work["openInterest"] = pd.to_numeric(work["openInterest"], errors="coerce").fillna(0.0)
                work = work[work["openInterest"] > 0]
                if len(work) == 0:
                    continue
                work["ratio"] = work["volume"] / work["openInterest"]
                hot = work.sort_values("ratio", ascending=False).head(1)
                if len(hot) > 0 and float(hot.iloc[0]["ratio"]) >= vol_oi_ratio_threshold:
                    options_alerts.append(
                        f"{chain_name} unusual activity: {hot.iloc[0]['contractSymbol']} volume/OI {float(hot.iloc[0]['ratio']):.2f}."
                    )

            earnings_map = get_earnings_dates([selected_opt_symbol])
            evt = earnings_map.get(selected_opt_symbol, None)
            evt_value = evt[0] if isinstance(evt, list) and len(evt) > 0 else evt
            evt_dt = pd.to_datetime(evt_value, errors='coerce')
            if not pd.isna(evt_dt):
                days_to_earnings = int((evt_dt.normalize() - pd.Timestamp.now().normalize()).days)
                if 0 <= days_to_earnings <= earnings_days_window:
                    options_alerts.append(f"Earnings expected in {days_to_earnings} day(s) on {evt_dt.strftime('%Y-%m-%d')}.")

            if options_alerts:
                for alert_msg in options_alerts[:8]:
                    st.warning(f"⚠️ {alert_msg}")
            else:
                st.success("✅ No options alerts triggered by current thresholds.")
        else:
            st.info("No options chain available for this symbol.")

if selected_view == "CSV Import":
    st.subheader("📥 CSV IMPORT (Yahoo-style)")
    st.caption("Import transactions or holdings snapshots for US stocks, Thai stocks, or Thai mutual funds. Extra columns are ignored.")
    st.caption("Auto-normalization: if Yahoo row hints indicate Thailand (THB/SET/Thailand), symbols like ADVANC are converted to ADVANC.BK.")

    templates = get_csv_templates()
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.download_button("⬇️ US Template", templates["us"], "template_us_transactions.csv", "text/csv")
    with col_t2:
        st.download_button("⬇️ Thai Stock Template", templates["thai_stock"], "template_thai_stock_transactions.csv", "text/csv")
    with col_t3:
        st.download_button("⬇️ Thai Fund Template", templates["thai_fund"], "template_thai_fund_transactions.csv", "text/csv")
    with col_t4:
        st.download_button("⬇️ Mixed Template", templates["mixed"], "template_mixed_transactions.csv", "text/csv")

    st.markdown("---")

    st.session_state["csv_default_asset_class"] = "Auto-detect"
    import_asset_class = st.selectbox(
        "Asset class mode",
        ["Auto-detect", "US Stock", "Thai Stock", "Mutual Fund"],
        key="csv_default_asset_class",
        help="Auto-detect uses per-row hints from Yahoo export (symbol/currency/market) and falls back to US Stock when ambiguous."
    )

    st.markdown("#### Step 1 — Upload")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader")

    if uploaded_csv is not None:
        parsed_df, parse_errors = parse_transactions_csv(uploaded_csv, import_asset_class)

        st.markdown("#### Step 2 — Validate & Preview")
        if parse_errors:
            st.warning(f"Found {len(parse_errors)} validation notes")
            with st.expander("Show validation notes"):
                for err in parse_errors[:50]:
                    st.write(f"- {err}")

        if parsed_df is not None and len(parsed_df) > 0:
            st.success(f"Parsed {len(parsed_df)} valid rows")
            st.info("Only relevant fields are extracted: instrument, action, quantity, price/cost basis, and date.")
            st.dataframe(parsed_df, hide_index=True, width="stretch")

            st.markdown("#### Dry Run Preview")
            preview_df, preview_summary = simulate_import_preview(parsed_df)

            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                st.metric("Total Rows", preview_summary["total"])
            with col_p2:
                st.metric("Will Import", preview_summary["import"])
            with col_p3:
                st.metric("Will Skip", preview_summary["skip"])
            with col_p4:
                st.metric("Will Fail", preview_summary["fail"])

            st.dataframe(preview_df, hide_index=True, width="stretch")

            st.markdown("#### Step 3 — Commit")
            st.caption("Imports valid rows and skips duplicates automatically.")
            skip_existing_duplicates = st.checkbox(
                "Skip previously imported duplicates",
                value=True,
                help="Turn off to re-apply rows (useful if portfolio was reset but history still exists).",
                key="csv_skip_existing_duplicates"
            )
            if st.button("Import Transactions", key="run_csv_import"):
                success_count = 0
                skipped_duplicates = 0
                failed_rows = []

                for _, import_row in parsed_df.iterrows():
                    ok, message = apply_import_transaction(import_row, skip_existing_duplicates=skip_existing_duplicates)
                    if ok:
                        success_count += 1
                    else:
                        if message == "Duplicate transaction skipped":
                            skipped_duplicates += 1
                        else:
                            failed_rows.append(f"Row {import_row['row_num']}: {message}")

                st.success(f"Imported {success_count} transactions")
                if skipped_duplicates > 0:
                    st.info(f"Skipped {skipped_duplicates} duplicate rows")
                    if success_count == 0:
                        st.caption("Tip: uncheck 'Skip previously imported duplicates' and import again to rebuild portfolio positions.")
                if failed_rows:
                    st.warning(f"Failed rows: {len(failed_rows)}")
                    for failed in failed_rows[:20]:
                        st.write(f"- {failed}")

                st.rerun()
        elif parsed_df is not None:
            st.info("No valid transaction rows found in CSV")

    with st.expander("CSV format examples"):
        st.markdown("""
        **Accepted aliases:**
        - Symbol: `symbol`, `ticker`, `code`
        - Fund code: `fund_code`, `fund`
        - Action: `action`, `type`, `transaction_type`
        - Quantity: `quantity`, `shares`, `units`
        - Price: `price`, `trade_price`, `nav`
        - Date: `trade_date`, `date`, `transaction_date`

        **US stocks sample:**
        ```csv
        symbol,action,quantity,price,date
        AAPL,BUY,10,190.50,2026-01-15
        AAPL,SELL,2,201.20,2026-02-01
        ```

        **Thai stocks sample:**
        ```csv
        symbol,action,quantity,price,date
        ADVANC.BK,BUY,50,220.00,2026-01-10
        TISCO.BK,BUY,100,98.50,2026-01-11
        ```

        **Thai funds sample:**
        ```csv
        fund_code,action,quantity,price,date,master
        SCBNDQ(E),BUY,1000,13.50,2026-01-05,QQQ
        SCBNDQ(E),SELL,100,13.95,2026-02-12,QQQ
        ```
        """)

if selected_view == "Calendar":
    st.subheader("🗓️ PORTFOLIO CALENDAR")
    st.caption("Plan and track earnings, dividends, rebalances & custom reminders")

    planner_col, remove_col = st.columns([2, 1], gap="large")
    with planner_col:
        with st.form("calendar_event_form"):
            st.markdown("**Add Upcoming Event**")
            evt_c1, evt_c2 = st.columns(2)
            with evt_c1:
                planned_date = st.date_input("Event Date", value=datetime.now().date(), key="planned_event_date")
                planned_type = st.selectbox(
                    "Event Type",
                    ["Reminder", "Earnings", "Macro", "Rebalance", "Dividend", "Risk Check", "Other"],
                    key="planned_event_type"
                )
            with evt_c2:
                planned_instrument = st.text_input("Instrument (optional)", key="planned_event_instrument")
                planned_title = st.text_input("Title", key="planned_event_title")
            planned_details = st.text_area("Details", key="planned_event_details")
            planned_submit = st.form_submit_button("Add Event")

            if planned_submit:
                clean_title = str(planned_title).strip()
                if not clean_title:
                    st.error("Event title is required.")
                else:
                    st.session_state.calendar_events.append({
                        "id": str(uuid.uuid4()),
                        "date": planned_date.strftime("%Y-%m-%d"),
                        "event_type": planned_type,
                        "title": clean_title,
                        "instrument": str(planned_instrument).strip().upper(),
                        "details": str(planned_details).strip(),
                        "status": "Planned",
                    })
                    st.session_state.calendar_events = st.session_state.calendar_events[-1200:]
                    save_calendar_events()
                    st.success("Event added")
                    st.rerun()

    with remove_col:
        st.markdown("**Manage Planned**")
        deletable = st.session_state.calendar_events[-200:]
        if deletable:
            delete_map = {
                f"{e.get('date', '')} · {e.get('event_type', 'Reminder')} · {e.get('title', '')}": e.get("id", "")
                for e in deletable
            }
            selected_delete = st.multiselect("Select events", options=list(delete_map.keys()), key="calendar_delete_events")
            if st.button("Delete Selected", key="calendar_delete_btn"):
                remove_ids = {delete_map[label] for label in selected_delete}
                st.session_state.calendar_events = [
                    e for e in st.session_state.calendar_events if e.get("id", "") not in remove_ids
                ]
                save_calendar_events()
                st.success("Selected events removed")
                st.rerun()
        else:
            st.caption("No planned events yet.")

    st.markdown("---")

    events = []
    today = pd.Timestamp.now().normalize()

    for event_item in st.session_state.calendar_events:
        parsed_event = pd.to_datetime(str(event_item.get("date", "")), errors='coerce')
        if pd.isna(parsed_event):
            continue
        events.append({
            "Date": parsed_event.normalize(),
            "Source": "Planned",
            "Event": str(event_item.get("event_type", "Reminder") or "Reminder"),
            "Instrument": str(event_item.get("instrument", "") or "").strip().upper(),
            "Details": str(event_item.get("title", "") or "").strip(),
            "Notes": str(event_item.get("details", "") or "").strip(),
        })

    # Upcoming earnings (US holdings)
    try:
        earnings_map = get_earnings_dates(df_us['Ticker'].tolist() if len(df_us) > 0 else [])
        for ticker, event_value in earnings_map.items():
            candidate = event_value[0] if isinstance(event_value, list) and len(event_value) > 0 else event_value
            parsed_event = pd.to_datetime(candidate, errors='coerce')
            if not pd.isna(parsed_event):
                event_date = parsed_event.normalize()
                events.append({
                    "Date": event_date,
                    "Source": "Market",
                    "Event": "Earnings",
                    "Instrument": ticker,
                    "Details": "Upcoming earnings date",
                    "Notes": "",
                })
    except Exception:
        pass

    # Recent and upcoming transaction activity
    for txn in st.session_state.transaction_history[-500:]:
        date_text = str(txn.get("date", "")).split(" ")[0]
        parsed_date = pd.to_datetime(date_text, errors='coerce')
        if pd.isna(parsed_date):
            continue
        event_date = parsed_date.normalize()
        events.append({
            "Date": event_date,
            "Source": "Journal",
            "Event": f"Transaction · {txn.get('type', 'N/A')}",
            "Instrument": txn.get("ticker", "N/A"),
            "Details": f"{float(txn.get('shares', 0) or 0):,.4f} @ {float(txn.get('price', 0) or 0):,.4f}",
            "Notes": str(txn.get("strategy", "") or "").strip(),
        })

    if events:
        df_events = pd.DataFrame(events).dropna(subset=["Date"])
        if len(df_events) > 0:
            df_events["Days"] = (df_events["Date"] - today).dt.days

            upcoming_events = df_events[df_events["Days"] >= 0].sort_values(["Date", "Source"]).head(80)
            recent_events = df_events[df_events["Days"] < 0].sort_values("Date", ascending=False).head(80)

            next_event_days = int(upcoming_events.iloc[0]["Days"]) if len(upcoming_events) > 0 else None
            next_event_label = str(upcoming_events.iloc[0]["Event"]) if len(upcoming_events) > 0 else "—"
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Upcoming (7d)", int((upcoming_events["Days"] <= 7).sum()) if len(upcoming_events) > 0 else 0)
            with k2:
                st.metric("Upcoming (30d)", int((upcoming_events["Days"] <= 30).sum()) if len(upcoming_events) > 0 else 0)
            with k3:
                st.metric("Next Event", next_event_label, delta=(f"in {next_event_days} day(s)" if next_event_days is not None else ""))

            cal_col1, cal_col2 = st.columns(2)
            with cal_col1:
                st.markdown("**Upcoming Timeline**")
                if len(upcoming_events) > 0:
                    st.dataframe(
                        upcoming_events[["Date", "Days", "Source", "Event", "Instrument", "Details", "Notes"]],
                        hide_index=True,
                        width='stretch'
                    )
                else:
                    st.info("No upcoming events.")

            with cal_col2:
                st.markdown("**Recent Activity**")
                if len(recent_events) > 0:
                    st.dataframe(
                        recent_events[["Date", "Days", "Source", "Event", "Instrument", "Details", "Notes"]],
                        hide_index=True,
                        width='stretch'
                    )
                else:
                    st.info("No recent events.")
        else:
            st.info("No calendar events available.")
    else:
        st.info("No calendar events available.")

if selected_view == "Trade Entry":
    st.subheader("📝 TRADE ENTRY")
    st.caption("Execute Buy/Sell/Dividend/Fee/Split actions in one dedicated workflow.")

    def _collect_journal_meta(prefix, transaction_type):
        with st.expander("📓 Journal Fields (Phase 1)", expanded=False):
            j1, j2 = st.columns(2)
            with j1:
                strategy = st.text_input("Strategy", key=f"{prefix}_strategy")
                session_name = st.selectbox(
                    "Session",
                    ["N/A", "Asia", "London", "New York", "Pre-market", "After-hours"],
                    key=f"{prefix}_session"
                )
                entry_window = st.selectbox(
                    "Entry Window",
                    ["N/A", "Open", "Mid-session", "Close"],
                    key=f"{prefix}_entry_window"
                )
            with j2:
                direction_default = "Long" if transaction_type == "Buy" else ("Short" if transaction_type == "Sell" else "N/A")
                direction = st.selectbox("Direction", ["N/A", "Long", "Short"], index=["N/A", "Long", "Short"].index(direction_default), key=f"{prefix}_direction")
                confidence = st.slider("Confidence (1-5)", min_value=1, max_value=5, value=3, key=f"{prefix}_confidence")
                followed_rules = st.checkbox("Followed Rules", value=True, key=f"{prefix}_followed_rules")
            risk_amount = st.number_input("Risk Amount", min_value=0.0, step=0.01, value=0.0, key=f"{prefix}_risk_amount")
            mental_state = st.selectbox(
                "Mental State",
                ["Neutral", "Calm", "Focused", "Anxious", "FOMO", "Revenge", "Overconfident"],
                key=f"{prefix}_mental_state"
            )
            journal_note = st.text_area("Journal Note", key=f"{prefix}_journal_note")

        return {
            "strategy": strategy,
            "session": session_name,
            "direction": direction,
            "followed_rules": followed_rules,
            "confidence": int(confidence),
            "risk_amount": float(risk_amount) if risk_amount > 0 else None,
            "entry_window": entry_window,
            "mental_state": mental_state,
            "journal_note": journal_note,
        }

    with st.expander("⚙️ Lot Method Policies", expanded=False):
        st.caption("Choose cost-basis method used for Sell realized P/L by asset class.")
        st.session_state.lot_method_policies["US Stock"] = st.selectbox(
            "US Stock method",
            ["FIFO", "LIFO", "AVERAGE"],
            index=["FIFO", "LIFO", "AVERAGE"].index(st.session_state.lot_method_policies.get("US Stock", "FIFO")),
            key="lot_policy_us"
        )
        st.session_state.lot_method_policies["Thai Stock"] = st.selectbox(
            "Thai Stock method",
            ["FIFO", "LIFO", "AVERAGE"],
            index=["FIFO", "LIFO", "AVERAGE"].index(st.session_state.lot_method_policies.get("Thai Stock", "FIFO")),
            key="lot_policy_thai"
        )
        st.session_state.lot_method_policies["Mutual Fund"] = st.selectbox(
            "Mutual Fund method",
            ["AVERAGE", "FIFO", "LIFO"],
            index=["AVERAGE", "FIFO", "LIFO"].index(st.session_state.lot_method_policies.get("Mutual Fund", "AVERAGE")),
            key="lot_policy_fund"
        )

    trade_col1, trade_col2 = st.columns(2, gap="large")

    with trade_col1:
        st.markdown("#### 🦅 US Equities")
        us_transaction_type = st.radio("US Transaction Type", ["Buy", "Sell", "Dividend", "Fee", "Split"], key="us_trans_type")

        with st.form("us_transaction_form"):
            us_ticker = st.text_input("Ticker Symbol (e.g., AAPL)", key="us_ticker")
            us_trade_date = st.date_input("Transaction Date", value=datetime.now().date(), key="us_trade_date")

            us_shares = 0.0
            us_price = 0.0
            us_cash_amount = 0.0
            us_split_ratio = 0.0

            if us_transaction_type in ["Buy", "Sell"]:
                us_shares = st.number_input("Shares", min_value=0.0, step=0.01, key="us_shares")
                us_price = st.number_input("Price per Share ($)", min_value=0.0, step=0.01, key="us_price")
            elif us_transaction_type in ["Dividend", "Fee"]:
                us_cash_amount = st.number_input("Cash Amount ($)", min_value=0.0, step=0.01, key="us_cash_amount")
            else:
                us_split_ratio = st.number_input("Split Ratio (new shares / old shares)", min_value=0.0001, step=0.0001, value=2.0, key="us_split_ratio")

            us_submit = st.form_submit_button(f"Add US {us_transaction_type}")
            us_journal_meta = _collect_journal_meta("us", us_transaction_type)

            if us_submit and us_ticker:
                us_ticker = us_ticker.strip().upper()

                if us_transaction_type in ["Dividend", "Fee"]:
                    if us_cash_amount <= 0:
                        st.error("❌ Amount must be greater than 0.")
                    else:
                        event_total = us_cash_amount if us_transaction_type == "Dividend" else -us_cash_amount
                        log_transaction(
                            us_ticker,
                            us_transaction_type,
                            0,
                            us_cash_amount,
                            "US Stock",
                            transaction_date=us_trade_date.strftime("%Y-%m-%d"),
                            total_override=event_total,
                            notes=f"{us_transaction_type} cash event",
                            journal_meta=us_journal_meta,
                        )
                        sign = "+" if us_transaction_type == "Dividend" else "-"
                        st.success(f"✅ Logged {us_transaction_type} for {us_ticker}: {sign}${us_cash_amount:.2f}")
                        st.rerun()

                elif us_transaction_type == "Split":
                    try:
                        ticker_idx = st.session_state.us_portfolio['Ticker'].index(us_ticker)
                        existing_shares = float(st.session_state.us_portfolio['Shares'][ticker_idx])
                        existing_avg_cost = float(st.session_state.us_portfolio['Avg_Cost'][ticker_idx])

                        if us_split_ratio <= 0:
                            st.error("❌ Split ratio must be greater than 0.")
                        else:
                            new_shares = existing_shares * us_split_ratio
                            new_avg_cost = existing_avg_cost / us_split_ratio
                            st.session_state.us_portfolio['Shares'][ticker_idx] = new_shares
                            st.session_state.us_portfolio['Avg_Cost'][ticker_idx] = new_avg_cost
                            lot_apply_split(us_ticker, "US Stock", "USD", us_split_ratio)
                            log_transaction(
                                us_ticker,
                                "Split",
                                us_split_ratio,
                                0,
                                "US Stock",
                                transaction_date=us_trade_date.strftime("%Y-%m-%d"),
                                total_override=0,
                                notes=f"Split ratio {us_split_ratio:g}:1",
                                journal_meta=us_journal_meta,
                            )
                            st.success(f"✅ Applied split for {us_ticker}: {us_split_ratio:g}:1")
                            st.rerun()
                    except ValueError:
                        st.error(f"❌ Cannot apply split for {us_ticker}. No existing position found.")

                elif us_shares > 0 and us_price > 0:
                    try:
                        ticker_idx = st.session_state.us_portfolio['Ticker'].index(us_ticker)
                        existing_shares = st.session_state.us_portfolio['Shares'][ticker_idx]
                        existing_avg_cost = st.session_state.us_portfolio['Avg_Cost'][ticker_idx]

                        if us_transaction_type == "Buy":
                            total_cost = (existing_shares * existing_avg_cost) + (us_shares * us_price)
                            new_shares = existing_shares + us_shares
                            new_avg_cost = total_cost / new_shares

                            st.session_state.us_portfolio['Shares'][ticker_idx] = new_shares
                            st.session_state.us_portfolio['Avg_Cost'][ticker_idx] = new_avg_cost
                            log_transaction(
                                us_ticker, "Buy", us_shares, us_price, "US Stock",
                                transaction_date=us_trade_date.strftime("%Y-%m-%d"),
                                journal_meta=us_journal_meta,
                            )
                            st.success(f"✅ Added {us_shares} shares of {us_ticker} @ ${us_price:.2f}")
                        else:
                            if us_shares > existing_shares:
                                st.error(f"❌ Cannot sell {us_shares} shares. Only {existing_shares} available.")
                            else:
                                log_transaction(
                                    us_ticker, "Sell", us_shares, us_price, "US Stock", existing_avg_cost,
                                    transaction_date=us_trade_date.strftime("%Y-%m-%d"),
                                    journal_meta=us_journal_meta,
                                )
                                new_shares = existing_shares - us_shares
                                if new_shares == 0:
                                    del st.session_state.us_portfolio['Ticker'][ticker_idx]
                                    del st.session_state.us_portfolio['Shares'][ticker_idx]
                                    del st.session_state.us_portfolio['Avg_Cost'][ticker_idx]
                                    st.success(f"✅ Sold all {us_ticker} shares")
                                else:
                                    st.session_state.us_portfolio['Shares'][ticker_idx] = new_shares
                                    st.success(f"✅ Sold {us_shares} shares of {us_ticker} @ ${us_price:.2f}")

                    except ValueError:
                        if us_transaction_type == "Buy":
                            st.session_state.us_portfolio['Ticker'].append(us_ticker)
                            st.session_state.us_portfolio['Shares'].append(us_shares)
                            st.session_state.us_portfolio['Avg_Cost'].append(us_price)
                            log_transaction(
                                us_ticker, "Buy", us_shares, us_price, "US Stock",
                                transaction_date=us_trade_date.strftime("%Y-%m-%d"),
                                journal_meta=us_journal_meta,
                            )
                            st.success(f"✅ Added new position: {us_shares} shares of {us_ticker} @ ${us_price:.2f}")
                        else:
                            st.error(f"❌ Cannot sell {us_ticker}. No existing position found.")

                    st.rerun()
                else:
                    st.error("❌ Enter valid amount/price values.")

    with trade_col2:
        st.markdown("#### 🏰 Thailand")
        thai_transaction_type = st.radio("Thai Transaction Type", ["Buy", "Sell", "Dividend", "Fee", "Split"], key="thai_trans_type")
        thai_vault_type = st.radio("Asset Type", ["Thai Stock", "Mutual Fund"], key="thai_vault_type")

        if thai_vault_type == "Thai Stock":
            with st.form("thai_stock_form"):
                thai_ticker = st.text_input("Ticker (e.g., TISCO.BK)", key="thai_ticker")
                thai_trade_date = st.date_input("Transaction Date", value=datetime.now().date(), key="thai_trade_date")
                thai_shares = 0.0
                thai_price = 0.0
                thai_cash_amount = 0.0
                thai_split_ratio = 0.0

                if thai_transaction_type in ["Buy", "Sell"]:
                    thai_shares = st.number_input("Shares", min_value=0.0, step=1.0, key="thai_shares")
                    thai_price = st.number_input("Price per Share (฿)", min_value=0.0, step=0.01, key="thai_price")
                elif thai_transaction_type in ["Dividend", "Fee"]:
                    thai_cash_amount = st.number_input("Cash Amount (฿)", min_value=0.0, step=0.01, key="thai_cash_amount")
                else:
                    thai_split_ratio = st.number_input("Split Ratio (new shares / old shares)", min_value=0.0001, step=0.0001, value=2.0, key="thai_split_ratio")

                thai_submit = st.form_submit_button(f"Add Thai {thai_transaction_type}")
                thai_journal_meta = _collect_journal_meta("thai_stock", thai_transaction_type)

                if thai_submit and thai_ticker:
                    thai_ticker = thai_ticker.strip().upper()
                    if not thai_ticker.endswith(".BK"):
                        thai_ticker = f"{thai_ticker}.BK"

                    if thai_transaction_type in ["Dividend", "Fee"]:
                        if thai_cash_amount <= 0:
                            st.error("❌ Amount must be greater than 0.")
                        else:
                            event_total = thai_cash_amount if thai_transaction_type == "Dividend" else -thai_cash_amount
                            log_transaction(
                                thai_ticker,
                                thai_transaction_type,
                                0,
                                thai_cash_amount,
                                "Thai Stock",
                                transaction_date=thai_trade_date.strftime("%Y-%m-%d"),
                                total_override=event_total,
                                notes=f"{thai_transaction_type} cash event",
                                journal_meta=thai_journal_meta,
                            )
                            sign = "+" if thai_transaction_type == "Dividend" else "-"
                            st.success(f"✅ Logged {thai_transaction_type} for {thai_ticker}: {sign}฿{thai_cash_amount:.2f}")
                            st.rerun()

                    elif thai_transaction_type == "Split":
                        try:
                            ticker_idx = st.session_state.thai_stocks['Ticker'].index(thai_ticker)
                            existing_shares = float(st.session_state.thai_stocks['Shares'][ticker_idx])
                            existing_avg_cost = float(st.session_state.thai_stocks['Avg_Cost'][ticker_idx])

                            if thai_split_ratio <= 0:
                                st.error("❌ Split ratio must be greater than 0.")
                            else:
                                new_shares = existing_shares * thai_split_ratio
                                new_avg_cost = existing_avg_cost / thai_split_ratio
                                st.session_state.thai_stocks['Shares'][ticker_idx] = new_shares
                                st.session_state.thai_stocks['Avg_Cost'][ticker_idx] = new_avg_cost
                                lot_apply_split(thai_ticker, "Thai Stock", "THB", thai_split_ratio)
                                log_transaction(
                                    thai_ticker,
                                    "Split",
                                    thai_split_ratio,
                                    0,
                                    "Thai Stock",
                                    transaction_date=thai_trade_date.strftime("%Y-%m-%d"),
                                    total_override=0,
                                    notes=f"Split ratio {thai_split_ratio:g}:1",
                                    journal_meta=thai_journal_meta,
                                )
                                st.success(f"✅ Applied split for {thai_ticker}: {thai_split_ratio:g}:1")
                                st.rerun()
                        except ValueError:
                            st.error(f"❌ Cannot apply split for {thai_ticker}. No existing position found.")

                    elif thai_shares > 0 and thai_price > 0:
                        try:
                            ticker_idx = st.session_state.thai_stocks['Ticker'].index(thai_ticker)
                            existing_shares = st.session_state.thai_stocks['Shares'][ticker_idx]
                            existing_avg_cost = st.session_state.thai_stocks['Avg_Cost'][ticker_idx]

                            if thai_transaction_type == "Buy":
                                total_cost = (existing_shares * existing_avg_cost) + (thai_shares * thai_price)
                                new_shares = existing_shares + thai_shares
                                new_avg_cost = total_cost / new_shares

                                st.session_state.thai_stocks['Shares'][ticker_idx] = new_shares
                                st.session_state.thai_stocks['Avg_Cost'][ticker_idx] = new_avg_cost
                                log_transaction(
                                    thai_ticker, "Buy", thai_shares, thai_price, "Thai Stock",
                                    transaction_date=thai_trade_date.strftime("%Y-%m-%d"),
                                    journal_meta=thai_journal_meta,
                                )
                                st.success(f"✅ Added {thai_shares} shares of {thai_ticker} @ ฿{thai_price:.2f}")
                            else:
                                if thai_shares > existing_shares:
                                    st.error(f"❌ Cannot sell {thai_shares} shares. Only {existing_shares} available.")
                                else:
                                    log_transaction(
                                        thai_ticker, "Sell", thai_shares, thai_price, "Thai Stock", existing_avg_cost,
                                        transaction_date=thai_trade_date.strftime("%Y-%m-%d"),
                                        journal_meta=thai_journal_meta,
                                    )
                                    new_shares = existing_shares - thai_shares
                                    if new_shares == 0:
                                        del st.session_state.thai_stocks['Ticker'][ticker_idx]
                                        del st.session_state.thai_stocks['Shares'][ticker_idx]
                                        del st.session_state.thai_stocks['Avg_Cost'][ticker_idx]
                                        st.success(f"✅ Sold all {thai_ticker} shares")
                                    else:
                                        st.session_state.thai_stocks['Shares'][ticker_idx] = new_shares
                                        st.success(f"✅ Sold {thai_shares} shares of {thai_ticker} @ ฿{thai_price:.2f}")

                        except ValueError:
                            if thai_transaction_type == "Buy":
                                st.session_state.thai_stocks['Ticker'].append(thai_ticker)
                                st.session_state.thai_stocks['Shares'].append(thai_shares)
                                st.session_state.thai_stocks['Avg_Cost'].append(thai_price)
                                log_transaction(
                                    thai_ticker, "Buy", thai_shares, thai_price, "Thai Stock",
                                    transaction_date=thai_trade_date.strftime("%Y-%m-%d"),
                                    journal_meta=thai_journal_meta,
                                )
                                st.success(f"✅ Added new position: {thai_shares} shares of {thai_ticker} @ ฿{thai_price:.2f}")
                            else:
                                st.error(f"❌ Cannot sell {thai_ticker}. No existing position found.")

                        st.rerun()
                    else:
                        st.error("❌ Enter valid amount/price values.")

        else:
            with st.form("thai_fund_form"):
                fund_code = st.text_input("Fund Code (e.g., SCBNDQ(E))", key="fund_code")
                fund_trade_date = st.date_input("Transaction Date", value=datetime.now().date(), key="fund_trade_date")
                fund_units = 0.0
                fund_price = 0.0
                fund_cash_amount = 0.0

                if thai_transaction_type in ["Buy", "Sell"]:
                    fund_units = st.number_input("Units", min_value=0.0, step=0.01, key="fund_units")
                    fund_price = st.number_input("NAV per Unit (฿)", min_value=0.0, step=0.0001, key="fund_price")
                elif thai_transaction_type in ["Dividend", "Fee"]:
                    fund_cash_amount = st.number_input("Cash Amount (฿)", min_value=0.0, step=0.01, key="fund_cash_amount")
                else:
                    st.caption("Split is available for Thai Stock only.")

                fund_master = st.selectbox("Master ETF", ["QQQ", "VOO", "VTI", "SOXX", "ICLN", "GLD", "^SET.BK", "N/A"], key="fund_master")
                fund_submit = st.form_submit_button(f"Add Fund {thai_transaction_type}")
                fund_journal_meta = _collect_journal_meta("thai_fund", thai_transaction_type)

                if fund_submit and fund_code:
                    fund_code = fund_code.strip().upper()

                    if thai_transaction_type in ["Dividend", "Fee"]:
                        if fund_cash_amount <= 0:
                            st.error("❌ Amount must be greater than 0.")
                        else:
                            event_total = fund_cash_amount if thai_transaction_type == "Dividend" else -fund_cash_amount
                            log_transaction(
                                fund_code,
                                thai_transaction_type,
                                0,
                                fund_cash_amount,
                                "Mutual Fund",
                                transaction_date=fund_trade_date.strftime("%Y-%m-%d"),
                                total_override=event_total,
                                notes=f"{thai_transaction_type} cash event",
                                journal_meta=fund_journal_meta,
                            )
                            sign = "+" if thai_transaction_type == "Dividend" else "-"
                            st.success(f"✅ Logged {thai_transaction_type} for {fund_code}: {sign}฿{fund_cash_amount:.2f}")
                            st.rerun()

                    elif thai_transaction_type == "Split":
                        st.error("❌ Split is only supported for Thai Stock positions.")

                    elif fund_units > 0 and fund_price > 0:
                        fund_idx = None
                        for i, fund in enumerate(st.session_state.vault_portfolio):
                            if fund['Code'] == fund_code:
                                fund_idx = i
                                break

                        if fund_idx is not None:
                            existing_units = st.session_state.vault_portfolio[fund_idx]['Units']
                            existing_cost = st.session_state.vault_portfolio[fund_idx]['Cost']
                            existing_master = str(st.session_state.vault_portfolio[fund_idx].get('Master', 'N/A') or 'N/A').strip().upper()

                            if thai_transaction_type == "Buy":
                                total_cost = (existing_units * existing_cost) + (fund_units * fund_price)
                                new_units = existing_units + fund_units
                                new_avg_cost = total_cost / new_units

                                st.session_state.vault_portfolio[fund_idx]['Units'] = new_units
                                st.session_state.vault_portfolio[fund_idx]['Cost'] = new_avg_cost
                                if existing_master in ["", "N/A"] and fund_master not in ["", "N/A"]:
                                    st.session_state.vault_portfolio[fund_idx]['Master'] = fund_master
                                log_transaction(
                                    fund_code, "Buy", fund_units, fund_price, "Mutual Fund",
                                    transaction_date=fund_trade_date.strftime("%Y-%m-%d"),
                                    journal_meta=fund_journal_meta,
                                    master=st.session_state.vault_portfolio[fund_idx].get('Master', fund_master),
                                )
                                st.success(f"✅ Added {fund_units} units of {fund_code} @ ฿{fund_price:.4f}")
                            else:
                                if fund_units > existing_units:
                                    st.error(f"❌ Cannot sell {fund_units} units. Only {existing_units} available.")
                                else:
                                    log_transaction(
                                        fund_code, "Sell", fund_units, fund_price, "Mutual Fund", existing_cost,
                                        transaction_date=fund_trade_date.strftime("%Y-%m-%d"),
                                        journal_meta=fund_journal_meta,
                                        master=existing_master,
                                    )
                                    new_units = existing_units - fund_units
                                    if new_units == 0:
                                        del st.session_state.vault_portfolio[fund_idx]
                                        st.success(f"✅ Sold all {fund_code} units")
                                    else:
                                        st.session_state.vault_portfolio[fund_idx]['Units'] = new_units
                                        st.success(f"✅ Sold {fund_units} units of {fund_code} @ ฿{fund_price:.4f}")
                        else:
                            if thai_transaction_type == "Buy":
                                st.session_state.vault_portfolio.append({
                                    "Code": fund_code,
                                    "Units": fund_units,
                                    "Cost": fund_price,
                                    "Master": fund_master
                                })
                                log_transaction(
                                    fund_code, "Buy", fund_units, fund_price, "Mutual Fund",
                                    transaction_date=fund_trade_date.strftime("%Y-%m-%d"),
                                    journal_meta=fund_journal_meta,
                                    master=fund_master,
                                )
                                st.success(f"✅ Added new fund: {fund_units} units of {fund_code} @ ฿{fund_price:.4f}")
                            else:
                                st.error(f"❌ Cannot sell {fund_code}. No existing position found.")

                        st.rerun()
                    else:
                        st.error("❌ Enter valid amount/price values.")

st.sidebar.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-nav-section">Config Status</div>', unsafe_allow_html=True)

sec_ok = bool(get_sec_api_keys())
news_ok = bool(get_newsapi_key())

st.sidebar.markdown(
    (
        '<div class="sidebar-status-row '
        + ('sidebar-status-ok' if sec_ok else 'sidebar-status-missing')
        + '"><span class="sidebar-status-dot"></span>Thai SEC API '
        + ('Configured' if sec_ok else 'Missing')
        + '</div>'
    ),
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    (
        '<div class="sidebar-status-row '
        + ('sidebar-status-ok' if news_ok else 'sidebar-status-missing')
        + '"><span class="sidebar-status-dot"></span>NewsAPI '
        + ('Configured' if news_ok else 'Missing')
        + '</div>'
    ),
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-nav-section">Data</div>', unsafe_allow_html=True)
if st.sidebar.button("🔄 Refresh market data", key="refresh_market_data", use_container_width=True):
    # Clear disk fund registry cache
    try:
        if _FUND_REGISTRY_CACHE_PATH.exists():
            _FUND_REGISTRY_CACHE_PATH.unlink()
    except Exception:
        pass
    for cached_fn in [
        fetch_quote_snapshot,
        fetch_fundamental_snapshot,
        fetch_options_snapshot,
        build_fund_registry,
        fetch_fund_nav_with_previous,
        fetch_fund_nav,
        fetch_news_for_ticker,
        _fetch_master_trends_cached,
        _fetch_latest_close_prices,
    ]:
        try:
            cached_fn.clear()
        except Exception:
            pass
    st.sidebar.success("Market data refreshed")
    st.rerun()

