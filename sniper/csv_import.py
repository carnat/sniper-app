"""CSV import parsing — Yahoo-style CSV transaction and holdings import.

Pure data processing functions with no Streamlit dependency.
"""

from datetime import datetime

import pandas as pd


def normalize_column_name(name: str) -> str:
    """Normalize a CSV column name for fuzzy matching."""
    return str(name).strip().lower().replace(" ", "").replace("_", "")


# Keep underscore-prefixed alias for backward compatibility
_normalize_column_name = normalize_column_name


def get_first_matching_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Find the first column in df that matches any of the given aliases."""
    normalized_map = {normalize_column_name(col): col for col in df.columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized_map:
            return normalized_map[key]
    return None


# Keep underscore-prefixed alias
_get_first_matching_column = get_first_matching_column


def normalize_asset_class(value: str) -> str | None:
    """Map raw asset class text to canonical form."""
    text = str(value).strip().lower()
    if text in ["us stock", "us", "stock", "equity", "us_equity"]:
        return "US Stock"
    if text in ["thai stock", "th stock", "thai", "thai_equity", "th_equity"]:
        return "Thai Stock"
    if text in ["mutual fund", "fund", "thai fund", "th fund", "thai_mutual_fund"]:
        return "Mutual Fund"
    return None


# Keep underscore-prefixed alias
_normalize_asset_class = normalize_asset_class


def looks_thai_market_hint(currency_value, market_value) -> bool:
    """Detect if a row is likely a Thai market instrument."""
    currency_text = str(currency_value).strip().upper() if currency_value is not None else ""
    market_text = str(market_value).strip().upper() if market_value is not None else ""
    return (
        currency_text == "THB"
        or "SET" in market_text
        or "BANGKOK" in market_text
        or "THAILAND" in market_text
        or "BK" in market_text
    )


# Keep underscore-prefixed alias
_looks_thai_market_hint = looks_thai_market_hint


def parse_numeric_value(value) -> float | None:
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


# Keep underscore-prefixed alias
_parse_numeric_value = parse_numeric_value


def parse_date_value(value) -> str | None:
    """Parse dates from Yahoo exports (supports YYYYMMDD, YYYYMMDD.0, and normal date strings)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if text == "":
        return None

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


# Keep underscore-prefixed alias
_parse_date_value = parse_date_value


def build_import_key(
    asset_class: str,
    action: str,
    symbol: str,
    fund_code: str,
    quantity: float,
    price: float,
    transaction_date: str,
) -> str:
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


def import_key_exists(import_key: str, transaction_history: list) -> bool:
    """Check whether an imported transaction key already exists in history.

    Args:
        import_key: The deterministic import key to check.
        transaction_history: List of transaction dicts to search.
    """
    if not import_key:
        return False
    for txn in transaction_history:
        if txn.get("import_key") == import_key:
            return True
    return False


def parse_transactions_csv(
    uploaded_file,
    default_asset_class: str,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Parse Yahoo-style CSV into canonical transaction rows.

    Supports:
    - transaction export: action + quantity + price + date
    - holdings export: symbol + shares + avg cost/cost basis
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, [f"Unable to read CSV: {exc}"]

    symbol_col = get_first_matching_column(df, ["symbol", "ticker", "code", "holding", "security"])
    fund_col = get_first_matching_column(df, ["fund_code", "fundcode", "fund"])
    action_col = get_first_matching_column(df, ["action", "type", "transaction_type", "transaction type"])
    yahoo_tx_type_col = get_first_matching_column(df, ["transaction_type", "transaction type"])
    quantity_col = get_first_matching_column(df, ["quantity", "shares", "units", "qty", "shares_owned", "shares owned"])
    price_col = get_first_matching_column(df, ["price", "trade_price", "fill_price", "avg_price", "nav", "purchase_price", "purchase price", "average_cost", "avg_cost", "avg cost", "cost", "current_price", "current price"])
    cost_basis_col = get_first_matching_column(df, ["cost_basis", "book_cost", "book cost", "total_cost", "cost basis"])
    date_col = get_first_matching_column(df, ["trade_date", "date", "transaction_date"])
    asset_col = get_first_matching_column(df, ["asset_class", "asset_type", "class"])
    currency_col = get_first_matching_column(df, ["currency", "curr", "trading_currency", "trading currency"])
    market_col = get_first_matching_column(df, ["market", "exchange", "country", "region"])
    master_col = get_first_matching_column(df, ["master", "master_etf", "benchmark"])

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
                if action_raw in ["", "NAN", "NONE"]:
                    action = "Buy"
                    action_inferred = True
                else:
                    row_errors.append(f"Row {row_num}: Unsupported action '{action_raw}'")
                    continue
        else:
            action = "Buy"
            action_inferred = True

        quantity = parse_numeric_value(row[quantity_col])
        if quantity is None or quantity <= 0:
            if action_inferred:
                continue
            row_errors.append(f"Row {row_num}: Invalid quantity")
            continue

        price = parse_numeric_value(row[price_col]) if price_col else None
        if (price is None or price <= 0) and cost_basis_col:
            cost_basis = parse_numeric_value(row[cost_basis_col])
            if cost_basis is not None and cost_basis > 0:
                price = cost_basis / quantity

        if price is None or price <= 0:
            if action_inferred:
                continue
            row_errors.append(f"Row {row_num}: Invalid price/cost basis")
            continue

        asset_class = fallback_asset_class
        if asset_col and pd.notna(row.get(asset_col)):
            mapped_asset = normalize_asset_class(row[asset_col])
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

        currency_hint = row.get(currency_col) if currency_col else None
        market_hint = row.get(market_col) if market_col else None
        if not fund_code and asset_col is None:
            if symbol.endswith(".BK") or looks_thai_market_hint(currency_hint, market_hint):
                asset_class = "Thai Stock"
            elif fallback_asset_class == "Mutual Fund":
                asset_class = "Mutual Fund"
            else:
                asset_class = fallback_asset_class

        if asset_class == "Thai Stock" and symbol and not symbol.endswith(".BK"):
            symbol = f"{symbol}.BK"

        date_value = datetime.now().strftime("%Y-%m-%d")
        if date_col and pd.notna(row.get(date_col)) and str(row[date_col]).strip():
            parsed_date = parse_date_value(row[date_col])
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


def get_csv_templates() -> dict[str, str]:
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
