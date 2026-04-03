"""Market data — quotes, fundamentals, options chains, stock data.

Functions that fetch live market data via yfinance. Caching decorators
are NOT included here — the Streamlit app adds @st.cache_data wrappers
around these functions for session-level caching.
"""

import pandas as pd
import yfinance as yf

from sniper.options import normalize_market_symbol

# Re-export for backward compatibility
_normalize_market_symbol = normalize_market_symbol


def get_base_symbol_universe(
    us_portfolio: dict,
    thai_stocks: dict,
    vault_portfolio: list,
    watchlists: dict | None = None,
) -> list[str]:
    """Build sorted list of all tracked symbols across portfolios and watchlists."""
    symbols = set()
    symbols.update([str(t).strip().upper() for t in us_portfolio.get("Ticker", []) if str(t).strip()])
    symbols.update([str(t).strip().upper() for t in thai_stocks.get("Ticker", []) if str(t).strip()])
    symbols.update([
        str(f.get("Master", "")).strip().upper()
        for f in vault_portfolio
        if str(f.get("Master", "")).strip() and str(f.get("Master", "")).strip().upper() != "N/A"
    ])
    if watchlists:
        for watch_symbols in watchlists.values():
            if isinstance(watch_symbols, list):
                symbols.update([str(t).strip().upper() for t in watch_symbols if str(t).strip()])
    return sorted(symbols)


# Keep underscore-prefixed alias
_get_base_symbol_universe = get_base_symbol_universe


def fetch_quote_snapshot(symbols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Fetch quote snapshot rows for a list of symbols."""
    rows = []
    for symbol in symbols:
        clean_symbol = normalize_market_symbol(symbol)
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


def fetch_fundamental_snapshot(symbol: str) -> dict:
    """Fetch Phase 2 fundamentals package for a symbol."""
    clean_symbol = normalize_market_symbol(symbol)
    empty_result = {
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
    if not clean_symbol:
        return empty_result

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

        def statement_to_frame(df):
            return df.T.reset_index().rename(columns={"index": "Period"}) if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()

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
        return empty_result


def fetch_options_snapshot(
    symbol: str,
    expiration: str | None = None,
) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """Fetch options chain scaffold data for a symbol."""
    clean_symbol = normalize_market_symbol(symbol)
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


def fetch_latest_close_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Fetch latest closing prices for a batch of tickers via yfinance."""
    if not tickers:
        return {}
    try:
        downloaded = yf.download(tickers, period="1d", progress=False)
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


# Keep underscore-prefixed alias
_fetch_latest_close_prices = fetch_latest_close_prices


def get_stock_data(portfolio_dict: dict, price_map: dict | None = None) -> pd.DataFrame:
    """Build portfolio DataFrame with live prices and P/L calculations.

    Args:
        portfolio_dict: {Ticker: [...], Shares: [...], Avg_Cost: [...]}.
        price_map: Optional pre-fetched {ticker: price} map. If None, fetches live.
    """
    df = pd.DataFrame(portfolio_dict)

    if len(df) == 0:
        return pd.DataFrame(columns=['Ticker', 'Shares', 'Avg_Cost', 'Live Price', 'Value', 'Cost Basis', 'P/L', 'P/L %'])

    tickers = df['Ticker'].tolist()
    if price_map is None:
        price_map = fetch_latest_close_prices(tuple(tickers))
    current_prices = [float(price_map.get(t, 0.0) or 0.0) for t in tickers]
    df['Live Price'] = current_prices
    df['Value'] = df['Shares'] * df['Live Price']
    df['Cost Basis'] = df['Shares'] * df['Avg_Cost']
    df['P/L'] = df['Value'] - df['Cost Basis']
    df['P/L %'] = df.apply(lambda r: (r['P/L'] / r['Cost Basis'] * 100) if r['Cost Basis'] != 0 else 0.0, axis=1)
    return df
