"""File persistence layer for Sniper OS.

All portfolio data is stored under the .streamlit/ directory.
Functions accept and return plain data structures — no Streamlit dependency.
"""

import json
import uuid
from pathlib import Path


def get_data_dir() -> Path:
    """Return the data directory for persistent storage."""
    return Path(".streamlit")


# --- Transaction History ---

def get_transactions_file_path() -> Path:
    """Local file path for persistent transaction history."""
    return get_data_dir() / "transactions.json"


def load_transaction_history() -> list:
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


def save_transaction_history(history: list) -> None:
    """Persist transaction history to local file."""
    try:
        file_path = get_transactions_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


# --- Alert State ---

def get_alert_state_file_path() -> Path:
    """Local file path for persistent alert trigger state."""
    return get_data_dir() / "alert_state.json"


def load_alert_state() -> dict:
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


def save_alert_state(state: dict) -> None:
    """Persist alert trigger state to local file."""
    try:
        file_path = get_alert_state_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


# --- Analytics Snapshots ---

def get_analytics_snapshots_file_path() -> Path:
    """Local file path for analytics snapshot history."""
    return get_data_dir() / "analytics_snapshots.json"


def load_analytics_snapshots() -> list:
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


def save_analytics_snapshots(snapshots: list) -> None:
    """Persist analytics snapshots to local file."""
    try:
        file_path = get_analytics_snapshots_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(snapshots, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


# --- Scenario Library ---

def get_scenario_library_file_path() -> Path:
    """Local file path for saved backtesting scenarios."""
    return get_data_dir() / "scenario_library.json"


def load_saved_scenarios() -> list:
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


def save_saved_scenarios(scenarios: list) -> None:
    """Persist saved scenarios to local file."""
    try:
        file_path = get_scenario_library_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(scenarios, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


# --- Watchlists ---

def get_watchlists_file_path() -> Path:
    """Local file path for persisted watchlists."""
    return get_data_dir() / "watchlists.json"


def load_watchlists() -> dict:
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
                        clean[name] = [
                            stripped.upper()
                            for s in symbols
                            if (stripped := str(s).strip())
                        ]
                    return clean
    except Exception:
        pass
    return {}


def save_watchlists(watchlists: dict) -> None:
    """Persist watchlists to local file."""
    try:
        file_path = get_watchlists_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(watchlists, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


# --- Options IV History ---

def get_options_iv_history_file_path() -> Path:
    """Local file path for observed ATM IV history used by IV rank/percentile."""
    return get_data_dir() / "options_iv_history.json"


def load_options_iv_history() -> dict:
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
                        clean[symbol] = clean_values[-400:]  # rolling 400-day window
                    return clean
    except Exception:
        pass
    return {}


def save_options_iv_history(history: dict) -> None:
    """Persist ATM IV history to local file."""
    try:
        file_path = get_options_iv_history_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


# --- Calendar Events ---

def get_calendar_events_file_path() -> Path:
    """Local file path for user-planned portfolio calendar events."""
    return get_data_dir() / "calendar_events.json"


def load_calendar_events() -> list:
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
                    return clean_events[-1200:]  # cap at ~3 years of daily events
    except Exception:
        pass
    return []


def save_calendar_events(events: list) -> None:
    """Persist user-planned calendar events to local file."""
    try:
        file_path = get_calendar_events_file_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(events, file, ensure_ascii=False, indent=2)
    except Exception:
        pass
