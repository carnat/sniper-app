"""Price alert engine — threshold detection with cooldown and deduplication."""

from datetime import datetime

import pandas as pd


def check_price_alerts(portfolio_dict, current_prices, asset_type, alert_state, threshold=5, cooldown_minutes=60):
    """Check holdings against threshold with dedupe + cooldown state.

    Parameters
    ----------
    portfolio_dict : dict with keys Ticker, Avg_Cost
    current_prices : list[float]
    asset_type : str
    alert_state : dict – mutable; updated in place and also returned
    threshold : float – percentage threshold for alerts
    cooldown_minutes : int

    Returns
    -------
    tuple(list[dict], dict) – (alerts, updated_alert_state)
    """
    alerts = []
    now = datetime.now()

    tickers = portfolio_dict.get("Ticker", [])
    avg_costs = portfolio_dict.get("Avg_Cost", [])
    currency_symbol = "$" if asset_type == "US Stock" else "฿"

    updated_state = dict(alert_state)

    for ticker, avg_cost, current_price in zip(tickers, avg_costs, current_prices):
        if avg_cost is None or avg_cost == 0:
            continue

        pct_change = ((current_price - avg_cost) / avg_cost) * 100
        direction = "up" if pct_change > 0 else "down"
        state_key = f"{asset_type}|{ticker}"
        previous_state = updated_state.get(state_key, {})
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

            updated_state[state_key] = {
                "triggered": True,
                "direction": direction,
                "last_alert_at": now.isoformat() if is_new else last_alert_at,
                "last_pct": pct_change,
            }

            alerts.append({
                "ticker": ticker,
                "change_pct": pct_change,
                "change_type": "📈 UP" if pct_change > 0 else "📉 DOWN",
                "current_price": current_price,
                "price_at_cost": avg_cost,
                "asset_type": asset_type,
                "currency_symbol": currency_symbol,
                "is_new": is_new,
            })
        else:
            updated_state[state_key] = {
                "triggered": False,
                "direction": None,
                "last_alert_at": last_alert_at,
                "last_pct": pct_change,
            }

    return alerts, updated_state

_check_price_alerts = check_price_alerts
