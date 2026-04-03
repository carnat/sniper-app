"""Options analytics — Black-Scholes Greeks, IV rank/percentile, ATM IV estimation.

Pure calculation functions with no Streamlit dependency. All functions that
previously relied on st.session_state now accept data as parameters.
"""

import math
from datetime import datetime
from statistics import NormalDist

import pandas as pd


def normalize_market_symbol(raw_text: str) -> str:
    """Normalize a market symbol to uppercase with whitespace stripped."""
    text = str(raw_text or "").strip().upper()
    if not text:
        return ""
    return text


# Keep underscore-prefixed alias for backward compatibility
_normalize_market_symbol = normalize_market_symbol


def record_atm_iv(symbol: str, atm_iv: float | None, iv_history: dict) -> dict:
    """Record one ATM IV observation per local day for a symbol.

    Args:
        symbol: Ticker symbol.
        atm_iv: Current ATM implied volatility value.
        iv_history: Mutable dict of {symbol: [{date, atm_iv}, ...]}.

    Returns:
        The updated iv_history dict.
    """
    if atm_iv is None:
        return iv_history
    clean_symbol = normalize_market_symbol(symbol)
    if not clean_symbol:
        return iv_history
    day_key = datetime.now().strftime("%Y-%m-%d")
    history = iv_history.get(clean_symbol, [])
    replaced = False
    for row in history:
        if str(row.get("date", "")) == day_key:
            row["atm_iv"] = float(atm_iv)
            replaced = True
            break
    if not replaced:
        history.append({"date": day_key, "atm_iv": float(atm_iv)})
    iv_history[clean_symbol] = history[-400:]
    return iv_history


# Keep underscore-prefixed alias for backward compatibility
_record_atm_iv = record_atm_iv


def compute_iv_rank_percentile(
    symbol: str,
    current_atm_iv: float | None,
    iv_history: dict,
) -> tuple:
    """Compute IV rank and percentile from locally observed ATM IV history.

    Args:
        symbol: Ticker symbol.
        current_atm_iv: Current ATM implied volatility.
        iv_history: Dict of {symbol: [{date, atm_iv}, ...]}.

    Returns:
        Tuple of (iv_rank, iv_percentile, history_length).
    """
    if current_atm_iv is None:
        return None, None, 0
    clean_symbol = normalize_market_symbol(symbol)
    history = iv_history.get(clean_symbol, [])
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


# Keep underscore-prefixed alias for backward compatibility
_compute_iv_rank_percentile = compute_iv_rank_percentile


def estimate_atm_iv(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    underlying_price: float | None,
) -> float | None:
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


# Keep underscore-prefixed alias for backward compatibility
_estimate_atm_iv = estimate_atm_iv


def annotate_black_scholes_greeks(
    chain_df: pd.DataFrame,
    option_type: str,
    underlying_price: float,
    expiry_text: str,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
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


# Keep underscore-prefixed alias for backward compatibility
_annotate_black_scholes_greeks = annotate_black_scholes_greeks
