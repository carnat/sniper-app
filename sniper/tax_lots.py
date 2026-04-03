"""Tax lot tracking — SQLite-backed FIFO, LIFO, and average-cost lot methods.

All lot operations use a local SQLite database at .streamlit/portfolio_lots.db.
No Streamlit dependency — all state is passed via parameters.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


def get_lot_db_path() -> Path:
    """Local SQLite file path for lot tracking."""
    return Path(".streamlit") / "portfolio_lots.db"


def init_lot_database() -> None:
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


def lot_record_buy(
    symbol: str,
    asset_type: str,
    currency: str,
    quantity: float,
    price: float,
    acquired_date: str | None = None,
    source: str = "BUY",
) -> None:
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


def lot_record_sell_fifo(
    symbol: str,
    asset_type: str,
    currency: str,
    quantity: float,
    sale_price: float,
    sale_date: str | None = None,
) -> float | None:
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


def lot_record_sell_lifo(
    symbol: str,
    asset_type: str,
    currency: str,
    quantity: float,
    sale_price: float,
    sale_date: str | None = None,
) -> float | None:
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


def lot_record_sell_average(
    symbol: str,
    asset_type: str,
    currency: str,
    quantity: float,
    sale_price: float,
    sale_date: str | None = None,
) -> float | None:
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


_LOT_METHOD_DEFAULTS = {
    "US Stock": "FIFO",
    "Thai Stock": "FIFO",
    "Mutual Fund": "AVERAGE",
}


def get_lot_method_for_asset(asset_type: str, policies: dict | None = None) -> str:
    """Resolve selected lot method policy by asset type.

    Args:
        asset_type: The asset class (e.g., "US Stock", "Mutual Fund").
        policies: Optional dict of {asset_type: method}. Falls back to defaults.
    """
    if policies is None:
        policies = {}
    return str(policies.get(asset_type, _LOT_METHOD_DEFAULTS.get(asset_type, "FIFO"))).upper()


def lot_apply_split(
    symbol: str,
    asset_type: str,
    currency: str,
    split_ratio: float,
) -> bool:
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


def seed_opening_lots_from_portfolios(
    us_portfolio: dict,
    thai_stocks: dict,
    vault_portfolio: list,
) -> None:
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

        for ticker, shares, avg_cost in zip(
            us_portfolio.get('Ticker', []),
            us_portfolio.get('Shares', []),
            us_portfolio.get('Avg_Cost', []),
        ):
            if float(shares) > 0:
                lot_record_buy(ticker, "US Stock", "USD", float(shares), float(avg_cost), acquired_date=today, source='OPENING')

        for ticker, shares, avg_cost in zip(
            thai_stocks.get('Ticker', []),
            thai_stocks.get('Shares', []),
            thai_stocks.get('Avg_Cost', []),
        ):
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
