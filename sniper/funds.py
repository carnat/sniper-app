"""Thai mutual fund data — SEC API registry, NAV fetching, master correlation."""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from ratelimit import limits, sleep_and_retry


# Mapping: Real fund names → SEC API abbreviations
FUND_API_MAPPING = {
    "SCBS&P500FUND(SSFA)": "SCBS&P500FUND(SSFA)",
    "SCBGOLDHE": "SCBGOLDHFUND",
    "SCB70SSF(SSFX)": "SCB70SSF",
    "SCBS&P500(E)": "SCBS&P500FUND(E)",
    "SCBS&P500-SSF": "SCBS&P500FUND(SSF)",
    "SCB70-SSFX": "SCB70SSF(SSFX)",
}

_FUND_REGISTRY_CACHE_PATH = Path(".streamlit/fund_registry_cache.json")
_FUND_REGISTRY_CACHE_MAX_AGE = 86400  # 24 hours


def normalize_fund_token(text):
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

# Backward-compat alias
_normalize_fund_token = normalize_fund_token


def resolve_fund_proj_id_and_class(fund_code, registry):
    """Resolve SEC proj_id + class suffix from imported fund code variants."""
    code_raw = str(fund_code or "").strip().upper()
    if not code_raw:
        return None, None

    api_code = FUND_API_MAPPING.get(code_raw, code_raw)

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
    registry_norm = {normalize_fund_token(k): v for k, v in registry.items()}
    for candidate in candidate_bases:
        norm = normalize_fund_token(candidate)
        if norm in registry_norm:
            return registry_norm[norm], class_suffix

    # 3) Fuzzy contains lookup for aliases like SCBS&P500 → SCBS&P500FUND
    best_match = None
    best_score = 10**9
    for candidate in candidate_bases:
        cand_norm = normalize_fund_token(candidate)
        if len(cand_norm) < 4:
            continue
        for reg_key, proj_id in registry.items():
            reg_norm = normalize_fund_token(reg_key)
            if cand_norm in reg_norm or reg_norm in cand_norm:
                score = abs(len(reg_norm) - len(cand_norm))
                if score < best_score:
                    best_score = score
                    best_match = proj_id
    if best_match:
        return best_match, class_suffix

    return None, class_suffix

_resolve_fund_proj_id_and_class = resolve_fund_proj_id_and_class


def get_sec_api_keys(secrets_keys=None):
    """Load SEC API keys from provided secrets list and/or environment variables."""
    keys = list(secrets_keys or [])
    keys = [k for k in keys if k]  # filter empty

    env_primary = str(os.getenv("SEC_API_PRIMARY_KEY", "")).strip()
    env_secondary = str(os.getenv("SEC_API_SECONDARY_KEY", "")).strip()
    if env_primary and env_primary not in keys:
        keys.append(env_primary)
    if env_secondary and env_secondary not in keys:
        keys.append(env_secondary)

    return keys

_get_sec_api_keys = get_sec_api_keys


@sleep_and_retry
@limits(calls=5, period=1)
def call_sec_api(url, api_keys=None):
    """Rate-limited API call to SEC endpoints."""
    if api_keys is None:
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


def load_disk_fund_registry():
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

_load_disk_fund_registry = load_disk_fund_registry


def save_disk_fund_registry(registry):
    """Persist fund registry to disk for fast cold starts."""
    try:
        _FUND_REGISTRY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FUND_REGISTRY_CACHE_PATH.write_text(json.dumps({"ts": time.time(), "registry": registry}))
    except Exception:
        pass

_save_disk_fund_registry = save_disk_fund_registry


def build_fund_registry(api_keys=None):
    """Build complete fund registry by querying all AMCs.

    Uses a 24-hour disk cache so cold starts are instant.
    Returns: {"SCBNDQ(E)": "M0000_2553", ...}
    """
    cached = load_disk_fund_registry()
    if cached:
        return cached

    fund_registry = {}

    try:
        amc_url = "https://api.sec.or.th/FundFactsheet/fund/amc"
        r_amc = call_sec_api(amc_url, api_keys=api_keys)

        if r_amc is None:
            return fund_registry

        amc_list = json.loads(r_amc.content)

        for amc in amc_list:
            unique_id = amc.get('unique_id')
            if not unique_id:
                continue

            fund_url = f"https://api.sec.or.th/FundFactsheet/fund/amc/{unique_id}"
            r_funds = call_sec_api(fund_url, api_keys=api_keys)

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

    if fund_registry:
        save_disk_fund_registry(fund_registry)

    return fund_registry

_build_fund_registry = build_fund_registry


def _extract_nav(nav_data, fund_class):
    """Helper to extract NAV from response data."""
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


def fetch_fund_nav_with_previous(proj_id, fund_class=None, api_keys=None):
    """Fetch latest NAV and previous day NAV for a fund by proj_id.

    Returns tuple: (last_val, previous_val) for day gain calculation.
    """
    try:
        candidate_dates = []
        now = datetime.now()
        days_back = 0
        while len(candidate_dates) < 7 and days_back < 14:
            d = now - timedelta(days=days_back)
            if d.weekday() < 5:
                candidate_dates.append(d.strftime("%Y-%m-%d"))
            days_back += 1

        found_dates = []

        for nav_date in candidate_dates:
            nav_url = f"https://api.sec.or.th/FundDailyInfo/{proj_id}/dailynav/{nav_date}"
            r = call_sec_api(nav_url, api_keys=api_keys)

            if r is not None and r.status_code == 200:
                nav_data = json.loads(r.content)
                last_val, previous_val = _extract_nav(nav_data, fund_class)

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

_fetch_fund_nav_with_previous = fetch_fund_nav_with_previous


def fetch_fund_nav(proj_id, fund_class=None, api_keys=None):
    """Fetch latest NAV for a fund by proj_id."""
    last_val, _ = fetch_fund_nav_with_previous(proj_id, fund_class, api_keys=api_keys)
    return last_val

_fetch_fund_nav = fetch_fund_nav


def fetch_master_trends(masters):
    """Fetch day-change % for master ETF symbols using yfinance."""
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

_fetch_master_trends = fetch_master_trends


def get_master_data(fund_list):
    """Build dict of master ETF day-change % for a fund list."""
    masters = [f.get('Master') for f in fund_list if f.get('Master') and f.get('Master') != 'N/A']
    if not masters:
        return {}
    return fetch_master_trends(tuple(masters))

_get_master_data = get_master_data


def get_fund_nav_by_code(fund_code, registry, api_keys=None):
    """Get NAV for a fund using the pre-built registry with fallback mechanisms."""
    proj_id, class_suffix = resolve_fund_proj_id_and_class(fund_code, registry)
    if proj_id:
        nav = fetch_fund_nav(proj_id, class_suffix, api_keys=api_keys)
        if nav > 0:
            return nav

        nav = fetch_fund_nav(proj_id, None, api_keys=api_keys)
        if nav > 0:
            return nav

        for alt_class in ['(E)', '(SSF)', '(SSFE)', '(SSFA)', '(SSFX)', '(A)']:
            nav = fetch_fund_nav(proj_id, alt_class, api_keys=api_keys)
            if nav > 0:
                return nav

    return 0.0


def get_fund_data(fund_list, registry, master_trends, fetch_nav_fn, get_nav_by_code_fn):
    """Assemble fund data rows (pure logic, no UI).

    Parameters
    ----------
    fund_list : list[dict]
    registry : dict  – fund registry from build_fund_registry
    master_trends : dict  – master ETF day-change map
    fetch_nav_fn : callable(proj_id, class_suffix) -> (current, prev)
    get_nav_by_code_fn : callable(code, registry) -> float
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    resolved = []
    for fund in fund_list:
        proj_id, class_suffix = resolve_fund_proj_id_and_class(fund.get('Code', ''), registry)
        resolved.append((fund, proj_id, class_suffix))

    nav_results = {}

    def _fetch_nav(idx, proj_id, class_suffix):
        if proj_id:
            return idx, fetch_nav_fn(proj_id, class_suffix)
        return idx, (0.0, 0.0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_fetch_nav, i, proj_id, class_suffix): i
            for i, (fund, proj_id, class_suffix) in enumerate(resolved)
        }
        for future in as_completed(futures):
            idx, nav_pair = future.result()
            nav_results[idx] = nav_pair

    data = []
    for i, (fund, proj_id, class_suffix) in enumerate(resolved):
        current_nav, prev_nav = nav_results.get(i, (0.0, 0.0))

        fund_day_gain = 0.0
        nav = current_nav
        if current_nav > 0 and prev_nav > 0:
            fund_day_gain = ((current_nav - prev_nav) / prev_nav) * 100

        if nav == 0:
            nav = get_nav_by_code_fn(fund['Code'], registry)
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

    df = pd.DataFrame(data)

    if len(df) == 0:
        return pd.DataFrame(columns=[
            'Code', 'Units', 'Cost', 'Last Price', 'Previous Price',
            'Fund Day Gain %', 'Master', 'Master Day Gain %',
            'Master vs Fund %', 'Cost Basis', 'Value', 'P/L', 'P/L %',
        ])

    df['Value'] = df['Units'] * df['Last Price']
    df['Cost Basis'] = df['Units'] * df['Cost']
    df['P/L'] = df['Value'] - df['Cost Basis']
    df['P/L %'] = (df['P/L'] / df['Cost Basis']) * 100
    df['Master vs Fund %'] = df['Fund Day Gain %'] - df['Master Day Gain %']
    df = df[[
        'Code', 'Units', 'Cost', 'Last Price', 'Previous Price',
        'Fund Day Gain %', 'Master', 'Master Day Gain %',
        'Master vs Fund %', 'Cost Basis', 'Value', 'P/L', 'P/L %',
    ]]

    return df
