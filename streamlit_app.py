import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from ratelimit import limits, sleep_and_retry

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
</style>
""", unsafe_allow_html=True)

st.title("🎯 SNIPER OS v1.2")
st.markdown("**HYBRID INTEL COMMAND CENTER**")
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
    notes=None
):
    """Log a transaction to history for P/L tracking"""
    from datetime import datetime
    
    currency = "USD" if asset_type == "US Stock" else "THB"
    if transaction_date is None:
        transaction_date = datetime.now().strftime("%Y-%m-%d")

    # Calculate realized P/L using lot tracking (FIFO), fallback to average-cost method.
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
        "realized_pl": realized_pl
    }

    if import_key:
        transaction["import_key"] = import_key
    if notes:
        transaction["notes"] = notes
    if lot_method_used:
        transaction["lot_method"] = lot_method_used
    
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
    df = df.rename(columns={"lot_method": "Lot Method"})
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
        qty = float(existing["qty"])
        avg_cost = float(existing["cost"])

        if txn_type == "Buy":
            total_cost = (qty * avg_cost) + (shares * price)
            new_qty = qty + shares
            if new_qty > 0:
                target[ticker] = {"qty": new_qty, "cost": total_cost / new_qty}
        else:
            new_qty = qty - shares
            if new_qty <= 1e-9:
                target.pop(ticker, None)
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
            "Master": "N/A"
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
}

# --- INTELLIGENCE ENGINES ---

# Rate Limiter: 5 calls per 1 second (SEC API limit)
# SEC API Subscription Keys
API_KEY_PRIMARY = "REDACTED_SEC_KEY_PRIMARY"
API_KEY_SECONDARY = "REDACTED_SEC_KEY_SECONDARY"

# Try primary key first, fallback to secondary
headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0',
    'Ocp-Apim-Subscription-Key': API_KEY_PRIMARY  # Will fallback to secondary if this fails
}

@sleep_and_retry
@limits(calls=5, period=1)
def call_sec_api(url):
    """Rate-limited API call to SEC endpoints"""
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response
        # For non-200 responses, still return None to trigger fallback
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def build_fund_registry():
    """
    Build complete fund registry by querying all AMCs.
    Returns: {"SCBNDQ(E)": "M0000_2553", ...}
    """
    fund_registry = {}
    
    try:
        # STEP 1: Get AMC list
        amc_url = "https://api.sec.or.th/FundFactsheet/fund/amc"
        r_amc = call_sec_api(amc_url)
        
        if r_amc is None:
            # st.warning("⚠️ SEC API authentication failed. Using cost basis for fund prices.")
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
                    # Create mapping: abbreviation -> proj_id
                    for fund in funds_list:
                        abbr = fund.get('proj_abbr_name')
                        proj_id = fund.get('proj_id')
                        if abbr and proj_id:
                            fund_registry[abbr] = proj_id
                except:
                    pass
    except Exception as e:
        pass
    
    return fund_registry

@st.cache_data(ttl=3600)
def fetch_fund_nav_with_previous(proj_id, fund_class=None):
    """
    Fetch latest NAV and previous day NAV for a fund by proj_id using FundDailyInfo endpoint.
    Format: /FundDailyInfo/{proj_id}/dailynav/{nav_date}
    Returns tuple: (last_val, previous_val) for day gain calculation.
    
    Strategy:
    1. Try to get previous_val from API response (if available)
    2. If not available, fetch two consecutive trading days separately
    
    Args:
        proj_id: Fund project ID (e.g., "M0311_2564")
        fund_class: Optional fund class suffix (e.g., "(E)", "(SSF)", "(SSFE)")
    """
    try:
        from datetime import datetime, timedelta
        
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
                
                # Fallback: get first record
                latest = nav_data[0]
                return (latest.get('last_val'), latest.get('previous_val'))
            elif isinstance(nav_data, dict):
                return (nav_data.get('last_val'), nav_data.get('previous_val'))
            return (None, None)
        
        # Try to find the most recent available trading day
        current_nav = None
        prev_nav = None
        found_dates = []
        
        for days_back in range(0, 15):
            nav_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            nav_url = f"https://api.sec.or.th/FundDailyInfo/{proj_id}/dailynav/{nav_date}"
            r = call_sec_api(nav_url)
            
            if r is not None and r.status_code == 200:
                nav_data = json.loads(r.content)
                last_val, previous_val = extract_nav(nav_data, fund_class)
                
                if last_val:
                    try:
                        last_price = float(last_val)
                        
                        # Check if API provides previous_val
                        if previous_val and str(previous_val).strip() and previous_val not in ['0', 0, '0.0', 0.0, '']:
                            prev_price = float(previous_val)
                            return (last_price, prev_price)
                        
                        # API doesn't provide previous_val, collect dates manually
                        found_dates.append((nav_date, last_price))
                        
                        # If we have 2 trading days, calculate from them
                        if len(found_dates) >= 2:
                            return (found_dates[0][1], found_dates[1][1])
                        
                    except Exception as e:
                        pass
        
        # If only found 1 day, return same value for both
        if len(found_dates) == 1:
            return (found_dates[0][1], 0.0)
            
    except Exception as e:
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
def get_master_data(fund_list):
    """
    Fetches live percent-change for master ETFs/indices from yfinance.
    Uses Ticker info for real-time up/down percentage from market.
    Handles US tickers (VOO, QQQ) and Thai indices (^SET.BK).
    Returns dict: {"VOO": 0.24, "^SET.BK": -0.12}
    """
    masters = list({f.get('Master') for f in fund_list if f.get('Master') and f.get('Master') != 'N/A'})
    master_map = {}

    if not masters:
        return master_map

    with st.spinner('🔮 Tracking Master ETFs...'):
        for m in masters:
            try:
                ticker = yf.Ticker(m)
                
                # Try to get regularMarketChangePercent from info (most reliable for live data)
                info = ticker.info
                if 'regularMarketChangePercent' in info and info['regularMarketChangePercent'] is not None:
                    master_map[m] = float(info['regularMarketChangePercent'])
                else:
                    # Fallback: Calculate from recent history
                    hist = ticker.history(period='2d')
                    if len(hist) >= 2:
                        last_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        pct_change = ((last_price - prev_price) / prev_price) * 100
                        master_map[m] = float(pct_change)
                    else:
                        master_map[m] = 0.0
            except Exception:
                master_map[m] = 0.0

    return master_map

def get_stock_data(portfolio_dict):
    df = pd.DataFrame(portfolio_dict)
    
    # Handle empty portfolio - return empty dataframe with correct columns
    if len(df) == 0:
        return pd.DataFrame(columns=['Ticker', 'Shares', 'Avg_Cost', 'Live Price', 'Value', 'Cost Basis', 'P/L', 'P/L %'])
    
    tickers = df['Ticker'].tolist()
    with st.spinner('Scanning US/Thai Stocks...'):
        try:
            data = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]
            if len(tickers) == 1: current_prices = [data.item()]
            else: current_prices = [data.get(t, 0.0) for t in tickers]
        except: current_prices = [0.0] * len(tickers)
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
    # Check if fund has an API mapping (real name differs from API name)
    api_code = fund_api_mapping.get(fund_code, fund_code)
    
    # Extract base fund name and class suffix
    # e.g., "SCBNDQ(E)" -> base="SCBNDQ", class="(E)"
    if '(' in api_code and api_code.endswith(')'):
        base_name = api_code[:api_code.index('(')]
        class_suffix = api_code[api_code.index('('):]
    else:
        base_name = api_code
        class_suffix = None
    
    # Look up proj_id by base name
    if base_name in registry:
        proj_id = registry[base_name]
        
        # Primary attempt with specified class suffix
        nav = fetch_fund_nav(proj_id, class_suffix)
        if nav > 0:
            return nav
        
        # Fallback 1: Try without class suffix (for navigation issues)
        nav = fetch_fund_nav(proj_id, None)
        if nav > 0:
            return nav
        
        # Fallback 2: Try common class suffixes (for funds like SCBGOLDHE that may have registry variations)
        for alt_class in ['(E)', '(SSF)', '(SSFE)', '(SSFA)', '(A)']:
            nav = fetch_fund_nav(proj_id, alt_class)
            if nav > 0:
                return nav
    
    return 0.0

def get_fund_data(fund_list):
    data = []
    bar = st.progress(0, text="Building Fund Registry & Fetching NAVs...")

    # Build registry once (cached)
    registry = build_fund_registry()

    # Fetch master ETF trends
    master_trends = get_master_data(fund_list)

    for i, fund in enumerate(fund_list):
        # Get current and previous NAV for day gain calculation
        api_code = fund_api_mapping.get(fund['Code'], fund['Code'])
        
        if '(' in api_code and api_code.endswith(')'):
            base_name = api_code[:api_code.index('(')]
            class_suffix = api_code[api_code.index('('):]
        else:
            base_name = api_code
            class_suffix = None
        
        # Use efficient single API call to get both last_val and previous_val
        fund_day_gain = 0.0
        nav = 0.0
        prev_nav = 0.0
        
        if base_name in registry:
            proj_id = registry[base_name]
            current_nav, previous_nav = fetch_fund_nav_with_previous(proj_id, class_suffix)
            
            nav = current_nav
            prev_nav = previous_nav
            
            # Calculate Fund Day Gain % using last_val and previous_val
            # If prev_nav is 0 or not available, try alternative approach or keep 0%
            if current_nav > 0 and prev_nav > 0:
                fund_day_gain = ((current_nav - prev_nav) / prev_nav) * 100
        
        # Fallback with class suffix variations if not found
        if nav == 0:
            nav = get_fund_nav_by_code(fund['Code'], registry)
        
        if nav == 0:
            nav = fund['Cost']  # Final fallback to cost basis

        row = fund.copy()
        row['Last Price'] = nav
        # Only show previous price if it's actually different from last price
        # If prev_nav is 0 or not available, show None (will display as empty in table)
        row['Previous Price'] = prev_nav if (prev_nav > 0 and prev_nav != nav) else None
        row['Fund Day Gain %'] = fund_day_gain

        # Attach master trend if available (rename to Master Day Gain %)
        master_ticker = fund.get('Master')
        if master_ticker and master_ticker != 'N/A':
            row['Master'] = master_ticker
            row['Master Day Gain %'] = master_trends.get(master_ticker, 0.0)
        else:
            row['Master'] = 'N/A'
            row['Master Day Gain %'] = 0.0

        data.append(row)
        bar.progress((i + 1) / len(fund_list))

    bar.empty()
    df = pd.DataFrame(data)
    
    # Handle empty portfolio - return empty dataframe with correct columns
    if len(df) == 0:
        return pd.DataFrame(columns=['Code', 'Units', 'Cost', 'Last Price', 'Previous Price', 'Fund Day Gain %', 'Master', 'Master Day Gain %', 'Master vs Fund %', 'Cost Basis', 'Value', 'P/L', 'P/L %'])
    
    df['Value'] = df['Units'] * df['Last Price']
    df['Cost Basis'] = df['Units'] * df['Cost']
    df['P/L'] = df['Value'] - df['Cost Basis']
    df['P/L %'] = (df['P/L'] / df['Cost Basis']) * 100
    
    # Calculate Master vs Fund correlation (difference between fund day performance and master day performance)
    df['Master vs Fund %'] = df['Fund Day Gain %'] - df['Master Day Gain %']
    
    # Reorder columns: Code, Units, Cost, Last Price, Previous Price, Fund Day Gain %, Master, Master Day Gain %, Master vs Fund %, Cost Basis, Value, P/L, P/L %
    df = df[['Code', 'Units', 'Cost', 'Last Price', 'Previous Price', 'Fund Day Gain %', 'Master', 'Master Day Gain %', 'Master vs Fund %', 'Cost Basis', 'Value', 'P/L', 'P/L %']]
    
    return df

# --- NEWS & ALERTS ENGINE ---

@st.cache_data(ttl=3600)
def fetch_news_for_ticker(ticker):
    """Fetch latest news for a ticker using NewsAPI (free tier)"""
    try:
        newsapi_key = st.secrets.get("news_alerts", {}).get("newsapi_key", "")
        if not newsapi_key:
            return []
        
        # Remove .BK extension for Thai stocks
        search_ticker = ticker.replace(".BK", "")
        
        url = f"https://newsapi.org/v2/everything?q={search_ticker}&sortBy=publishedAt&language=en&pageSize=5"
        headers = {"Authorization": newsapi_key}
        
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            return articles
    except Exception as e:
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
        if action == "Buy":
            total_cost = (existing_units * existing_cost) + (quantity * price)
            new_units = existing_units + quantity
            st.session_state.vault_portfolio[fund_idx]['Units'] = new_units
            st.session_state.vault_portfolio[fund_idx]['Cost'] = total_cost / new_units
            log_transaction(fund_code, "Buy", quantity, price, "Mutual Fund", transaction_date=transaction_date, import_key=import_key)
            return True, "ok"

        if quantity > existing_units:
            return False, f"Insufficient units for {fund_code}"
        log_transaction(fund_code, "Sell", quantity, price, "Mutual Fund", existing_cost, transaction_date=transaction_date, import_key=import_key)
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
        log_transaction(fund_code, "Buy", quantity, price, "Mutual Fund", transaction_date=transaction_date, import_key=import_key)
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

st.markdown("""---""")
st.markdown("")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🦅 US ATTACK", "🏰 THAI VAULT", "📊 ANALYTICS", "📰 NEWS WATCHTOWER", "📜 TRANSACTION HISTORY", "📥 CSV IMPORT", "🗓️ CALENDAR"])

def color_pl(val): return f'color: {"#3fb950" if val > 0 else "#f85149"}; font-weight: bold;'

with tab1:
    st.subheader("US Equities Portfolio")
    st.dataframe(df_us.style.applymap(color_pl, subset=['P/L %'])
                 .format({"Shares":"{:,.2f}", "Avg_Cost":"${:,.2f}", "Live Price":"${:,.2f}", "Value":"${:,.2f}", "Cost Basis":"${:,.2f}", "P/L":"${:,.2f}", "P/L %":"{:.2f}%"}),
                 hide_index=True, width='stretch')

with tab2:
    st.subheader("Mutual Funds (SEC Direct Data)")
    st.caption("*Real-time NAV from api.sec.or.th | Master vs Fund % shows daily correlation with master ETF*")
    # Calculate dynamic height: header (40px) + rows (35px each) + padding (10px)
    table_height = min(40 + len(df_vault) * 35 + 10, 800)
    st.dataframe(df_vault.style.applymap(color_pl, subset=['P/L %', 'Fund Day Gain %', 'Master Day Gain %', 'Master vs Fund %'])
                 .format({
                     "Units":"{:,.2f}",
                     "Cost":"฿{:,.4f}",
                     "Last Price":"฿{:,.4f}",
                     "Previous Price":"฿{:,.4f}",
                     "Fund Day Gain %":"{:+.2f}%",
                     "Master Day Gain %":"{:+.2f}%",
                     "Master vs Fund %":"{:+.2f}%",
                     "Cost Basis":"฿{:,.2f}",
                     "Value":"฿{:,.2f}",
                     "P/L":"฿{:,.2f}",
                     "P/L %":"{:+.2f}%"
                 }),
                 hide_index=True, width='stretch', height=table_height)
    
    st.markdown("---")
    st.subheader("Thai Equities")
    st.dataframe(df_thai.style.applymap(color_pl, subset=['P/L %'])
                 .format({"Shares":"{:,.2f}", "Avg_Cost":"฿{:,.2f}", "Live Price":"฿{:,.2f}", "Value":"฿{:,.2f}", "Cost Basis":"฿{:,.2f}", "P/L":"฿{:,.2f}", "P/L %":"{:.2f}%"}),
                 hide_index=True, width='stretch')

with tab3:
    st.subheader("Performance Analysis")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**US Equities P/L %**")
        st.bar_chart(df_us.set_index('Ticker')['P/L %'])
    
    with col_b:
        st.markdown("**Mutual Funds P/L %**")
        st.bar_chart(df_vault.set_index('Code')['P/L %'])

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

with tab4:
    st.subheader("📰 NEWS WATCHTOWER")
    st.caption("Real-time news and alerts for your holdings")
    
    # Get all tickers from portfolios
    all_tickers = us_portfolio.get("Ticker", []) + thai_stocks.get("Ticker", [])
    
    if not all_tickers:
        st.info("📌 Add holdings to see news and alerts for your positions")
    else:
        # Check if NewsAPI key is configured
        newsapi_key = st.secrets.get("news_alerts", {}).get("newsapi_key", "")
        
        if not newsapi_key:
            st.warning("⚠️ NewsAPI key not configured. Add your free key from https://newsapi.org/ to `.streamlit/secrets.toml` under `[news_alerts]` → `newsapi_key`")
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
                    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

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

with tab5:
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
                "Asset Type", "Currency", "Lot Method", "Realized P/L Display"
            ]],
            hide_index=True,
            use_container_width=True
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
        st.info("📌 No transactions yet. Add buy/sell transactions using the sidebar to track your realized P/L.")
    
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

with tab6:
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
            st.dataframe(parsed_df, hide_index=True, use_container_width=True)

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

            st.dataframe(preview_df, hide_index=True, use_container_width=True)

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

with tab7:
    st.subheader("🗓️ PORTFOLIO CALENDAR")

    events = []
    today = pd.Timestamp.now().normalize()

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
                    "Event": "Earnings",
                    "Instrument": ticker,
                    "Details": "Upcoming earnings date",
                })
    except Exception:
        pass

    # Recent and upcoming transaction activity
    for txn in st.session_state.transaction_history[-300:]:
        date_text = str(txn.get("date", "")).split(" ")[0]
        parsed_date = pd.to_datetime(date_text, errors='coerce')
        if pd.isna(parsed_date):
            continue
        event_date = parsed_date.normalize()
        events.append({
            "Date": event_date,
            "Event": f"Transaction · {txn.get('type', 'N/A')}",
            "Instrument": txn.get("ticker", "N/A"),
            "Details": f"{float(txn.get('shares', 0) or 0):,.4f} @ {float(txn.get('price', 0) or 0):,.4f}",
        })

    if events:
        df_events = pd.DataFrame(events).dropna(subset=["Date"])
        if len(df_events) > 0:
            df_events["Days"] = (df_events["Date"] - today).dt.days

            upcoming_events = df_events[df_events["Days"] >= 0].sort_values("Date").head(40)
            recent_events = df_events[df_events["Days"] < 0].sort_values("Date", ascending=False).head(40)

            cal_col1, cal_col2 = st.columns(2)
            with cal_col1:
                st.markdown("**Upcoming**")
                if len(upcoming_events) > 0:
                    st.dataframe(
                        upcoming_events[["Date", "Days", "Event", "Instrument", "Details"]],
                        hide_index=True,
                        width='stretch'
                    )
                else:
                    st.info("No upcoming events.")

            with cal_col2:
                st.markdown("**Recent Activity**")
                if len(recent_events) > 0:
                    st.dataframe(
                        recent_events[["Date", "Days", "Event", "Instrument", "Details"]],
                        hide_index=True,
                        width='stretch'
                    )
                else:
                    st.info("No recent events.")
        else:
            st.info("No calendar events available.")
    else:
        st.info("No calendar events available.")

# --- TRANSACTION SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.header("💰 ADD TRANSACTIONS")

with st.sidebar.expander("⚙️ Lot Method Policies", expanded=False):
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

# US VAULT TRANSACTIONS
st.sidebar.subheader("🦅 US Vault")
us_transaction_type = st.sidebar.radio("US Transaction Type", ["Buy", "Sell", "Dividend", "Fee", "Split"], key="us_trans_type")

with st.sidebar.form("us_transaction_form"):
    us_ticker = st.text_input("Ticker Symbol (e.g., AAPL)", key="us_ticker")

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

    if us_submit and us_ticker:
        us_ticker = us_ticker.strip().upper()

        if us_transaction_type in ["Dividend", "Fee"]:
            if us_cash_amount <= 0:
                st.sidebar.error("❌ Amount must be greater than 0.")
            else:
                event_total = us_cash_amount if us_transaction_type == "Dividend" else -us_cash_amount
                log_transaction(
                    us_ticker,
                    us_transaction_type,
                    0,
                    us_cash_amount,
                    "US Stock",
                    total_override=event_total,
                    notes=f"{us_transaction_type} cash event"
                )
                sign = "+" if us_transaction_type == "Dividend" else "-"
                st.sidebar.success(f"✅ Logged {us_transaction_type} for {us_ticker}: {sign}${us_cash_amount:.2f}")
                st.rerun()

        elif us_transaction_type == "Split":
            try:
                ticker_idx = st.session_state.us_portfolio['Ticker'].index(us_ticker)
                existing_shares = float(st.session_state.us_portfolio['Shares'][ticker_idx])
                existing_avg_cost = float(st.session_state.us_portfolio['Avg_Cost'][ticker_idx])

                if us_split_ratio <= 0:
                    st.sidebar.error("❌ Split ratio must be greater than 0.")
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
                        total_override=0,
                        notes=f"Split ratio {us_split_ratio:g}:1"
                    )
                    st.sidebar.success(f"✅ Applied split for {us_ticker}: {us_split_ratio:g}:1")
                    st.rerun()
            except ValueError:
                st.sidebar.error(f"❌ Cannot apply split for {us_ticker}. No existing position found.")

        elif us_shares > 0 and us_price > 0:
        # Find if ticker exists
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
                    log_transaction(us_ticker, "Buy", us_shares, us_price, "US Stock")
                    st.sidebar.success(f"✅ Added {us_shares} shares of {us_ticker} @ ${us_price:.2f}")
                else:
                    if us_shares > existing_shares:
                        st.sidebar.error(f"❌ Cannot sell {us_shares} shares. Only {existing_shares} available.")
                    else:
                        log_transaction(us_ticker, "Sell", us_shares, us_price, "US Stock", existing_avg_cost)
                        new_shares = existing_shares - us_shares
                        if new_shares == 0:
                            del st.session_state.us_portfolio['Ticker'][ticker_idx]
                            del st.session_state.us_portfolio['Shares'][ticker_idx]
                            del st.session_state.us_portfolio['Avg_Cost'][ticker_idx]
                            st.sidebar.success(f"✅ Sold all {us_ticker} shares")
                        else:
                            st.session_state.us_portfolio['Shares'][ticker_idx] = new_shares
                            st.sidebar.success(f"✅ Sold {us_shares} shares of {us_ticker} @ ${us_price:.2f}")

            except ValueError:
                if us_transaction_type == "Buy":
                    st.session_state.us_portfolio['Ticker'].append(us_ticker)
                    st.session_state.us_portfolio['Shares'].append(us_shares)
                    st.session_state.us_portfolio['Avg_Cost'].append(us_price)
                    log_transaction(us_ticker, "Buy", us_shares, us_price, "US Stock")
                    st.sidebar.success(f"✅ Added new position: {us_shares} shares of {us_ticker} @ ${us_price:.2f}")
                else:
                    st.sidebar.error(f"❌ Cannot sell {us_ticker}. No existing position found.")

            st.rerun()
        else:
            st.sidebar.error("❌ Enter valid amount/price values.")

st.sidebar.markdown("---")

# THAI VAULT TRANSACTIONS
st.sidebar.subheader("🏰 Thai Vault")
thai_transaction_type = st.sidebar.radio("Thai Transaction Type", ["Buy", "Sell", "Dividend", "Fee", "Split"], key="thai_trans_type")

thai_vault_type = st.sidebar.radio("Asset Type", ["Thai Stock", "Mutual Fund"], key="thai_vault_type")

if thai_vault_type == "Thai Stock":
    with st.sidebar.form("thai_stock_form"):
        thai_ticker = st.text_input("Ticker (e.g., TISCO.BK)", key="thai_ticker")
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

        if thai_submit and thai_ticker:
            thai_ticker = thai_ticker.strip().upper()
            if not thai_ticker.endswith(".BK"):
                thai_ticker = f"{thai_ticker}.BK"

            if thai_transaction_type in ["Dividend", "Fee"]:
                if thai_cash_amount <= 0:
                    st.sidebar.error("❌ Amount must be greater than 0.")
                else:
                    event_total = thai_cash_amount if thai_transaction_type == "Dividend" else -thai_cash_amount
                    log_transaction(
                        thai_ticker,
                        thai_transaction_type,
                        0,
                        thai_cash_amount,
                        "Thai Stock",
                        total_override=event_total,
                        notes=f"{thai_transaction_type} cash event"
                    )
                    sign = "+" if thai_transaction_type == "Dividend" else "-"
                    st.sidebar.success(f"✅ Logged {thai_transaction_type} for {thai_ticker}: {sign}฿{thai_cash_amount:.2f}")
                    st.rerun()

            elif thai_transaction_type == "Split":
                try:
                    ticker_idx = st.session_state.thai_stocks['Ticker'].index(thai_ticker)
                    existing_shares = float(st.session_state.thai_stocks['Shares'][ticker_idx])
                    existing_avg_cost = float(st.session_state.thai_stocks['Avg_Cost'][ticker_idx])

                    if thai_split_ratio <= 0:
                        st.sidebar.error("❌ Split ratio must be greater than 0.")
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
                            total_override=0,
                            notes=f"Split ratio {thai_split_ratio:g}:1"
                        )
                        st.sidebar.success(f"✅ Applied split for {thai_ticker}: {thai_split_ratio:g}:1")
                        st.rerun()
                except ValueError:
                    st.sidebar.error(f"❌ Cannot apply split for {thai_ticker}. No existing position found.")

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
                        log_transaction(thai_ticker, "Buy", thai_shares, thai_price, "Thai Stock")
                        st.sidebar.success(f"✅ Added {thai_shares} shares of {thai_ticker} @ ฿{thai_price:.2f}")
                    else:
                        if thai_shares > existing_shares:
                            st.sidebar.error(f"❌ Cannot sell {thai_shares} shares. Only {existing_shares} available.")
                        else:
                            log_transaction(thai_ticker, "Sell", thai_shares, thai_price, "Thai Stock", existing_avg_cost)
                            new_shares = existing_shares - thai_shares
                            if new_shares == 0:
                                del st.session_state.thai_stocks['Ticker'][ticker_idx]
                                del st.session_state.thai_stocks['Shares'][ticker_idx]
                                del st.session_state.thai_stocks['Avg_Cost'][ticker_idx]
                                st.sidebar.success(f"✅ Sold all {thai_ticker} shares")
                            else:
                                st.session_state.thai_stocks['Shares'][ticker_idx] = new_shares
                                st.sidebar.success(f"✅ Sold {thai_shares} shares of {thai_ticker} @ ฿{thai_price:.2f}")

                except ValueError:
                    if thai_transaction_type == "Buy":
                        st.session_state.thai_stocks['Ticker'].append(thai_ticker)
                        st.session_state.thai_stocks['Shares'].append(thai_shares)
                        st.session_state.thai_stocks['Avg_Cost'].append(thai_price)
                        log_transaction(thai_ticker, "Buy", thai_shares, thai_price, "Thai Stock")
                        st.sidebar.success(f"✅ Added new position: {thai_shares} shares of {thai_ticker} @ ฿{thai_price:.2f}")
                    else:
                        st.sidebar.error(f"❌ Cannot sell {thai_ticker}. No existing position found.")

                st.rerun()
            else:
                st.sidebar.error("❌ Enter valid amount/price values.")

else:  # Mutual Fund
    with st.sidebar.form("thai_fund_form"):
        fund_code = st.text_input("Fund Code (e.g., SCBNDQ(E))", key="fund_code")
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

        if fund_submit and fund_code:
            fund_code = fund_code.strip().upper()

            if thai_transaction_type in ["Dividend", "Fee"]:
                if fund_cash_amount <= 0:
                    st.sidebar.error("❌ Amount must be greater than 0.")
                else:
                    event_total = fund_cash_amount if thai_transaction_type == "Dividend" else -fund_cash_amount
                    log_transaction(
                        fund_code,
                        thai_transaction_type,
                        0,
                        fund_cash_amount,
                        "Mutual Fund",
                        total_override=event_total,
                        notes=f"{thai_transaction_type} cash event"
                    )
                    sign = "+" if thai_transaction_type == "Dividend" else "-"
                    st.sidebar.success(f"✅ Logged {thai_transaction_type} for {fund_code}: {sign}฿{fund_cash_amount:.2f}")
                    st.rerun()

            elif thai_transaction_type == "Split":
                st.sidebar.error("❌ Split is only supported for Thai Stock positions.")

            elif fund_units > 0 and fund_price > 0:
            # Find if fund exists
                fund_idx = None
                for i, fund in enumerate(st.session_state.vault_portfolio):
                    if fund['Code'] == fund_code:
                        fund_idx = i
                        break

                if fund_idx is not None:
                    existing_units = st.session_state.vault_portfolio[fund_idx]['Units']
                    existing_cost = st.session_state.vault_portfolio[fund_idx]['Cost']

                    if thai_transaction_type == "Buy":
                        total_cost = (existing_units * existing_cost) + (fund_units * fund_price)
                        new_units = existing_units + fund_units
                        new_avg_cost = total_cost / new_units

                        st.session_state.vault_portfolio[fund_idx]['Units'] = new_units
                        st.session_state.vault_portfolio[fund_idx]['Cost'] = new_avg_cost
                        log_transaction(fund_code, "Buy", fund_units, fund_price, "Mutual Fund")
                        st.sidebar.success(f"✅ Added {fund_units} units of {fund_code} @ ฿{fund_price:.4f}")
                    else:
                        if fund_units > existing_units:
                            st.sidebar.error(f"❌ Cannot sell {fund_units} units. Only {existing_units} available.")
                        else:
                            log_transaction(fund_code, "Sell", fund_units, fund_price, "Mutual Fund", existing_cost)
                            new_units = existing_units - fund_units
                            if new_units == 0:
                                del st.session_state.vault_portfolio[fund_idx]
                                st.sidebar.success(f"✅ Sold all {fund_code} units")
                            else:
                                st.session_state.vault_portfolio[fund_idx]['Units'] = new_units
                                st.sidebar.success(f"✅ Sold {fund_units} units of {fund_code} @ ฿{fund_price:.4f}")
                else:
                    if thai_transaction_type == "Buy":
                        st.session_state.vault_portfolio.append({
                            "Code": fund_code,
                            "Units": fund_units,
                            "Cost": fund_price,
                            "Master": fund_master
                        })
                        log_transaction(fund_code, "Buy", fund_units, fund_price, "Mutual Fund")
                        st.sidebar.success(f"✅ Added new fund: {fund_units} units of {fund_code} @ ฿{fund_price:.4f}")
                    else:
                        st.sidebar.error(f"❌ Cannot sell {fund_code}. No existing position found.")

                st.rerun()
            else:
                st.sidebar.error("❌ Enter valid amount/price values.")

