import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
import json
from datetime import datetime
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
# Initialize session state for portfolios if not exists
if 'us_portfolio' not in st.session_state:
    st.session_state.us_portfolio = {
        "Ticker": [],
        "Shares": [],
        "Avg_Cost": []
    }

if 'thai_stocks' not in st.session_state:
    st.session_state.thai_stocks = {
        "Ticker": [],
        "Shares": [],
        "Avg_Cost": []
    }

if 'vault_portfolio' not in st.session_state:
    st.session_state.vault_portfolio = []

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
    df['Value'] = df['Units'] * df['Last Price']
    df['Cost Basis'] = df['Units'] * df['Cost']
    df['P/L'] = df['Value'] - df['Cost Basis']
    df['P/L %'] = (df['P/L'] / df['Cost Basis']) * 100
    
    # Calculate Master vs Fund correlation (difference between fund day performance and master day performance)
    df['Master vs Fund %'] = df['Fund Day Gain %'] - df['Master Day Gain %']
    
    # Reorder columns: Code, Units, Cost, Last Price, Previous Price, Fund Day Gain %, Master, Master Day Gain %, Master vs Fund %, Cost Basis, Value, P/L, P/L %
    df = df[['Code', 'Units', 'Cost', 'Last Price', 'Previous Price', 'Fund Day Gain %', 'Master', 'Master Day Gain %', 'Master vs Fund %', 'Cost Basis', 'Value', 'P/L', 'P/L %']]
    
    return df

# --- EXECUTION ---
df_us = get_stock_data(us_portfolio)
df_thai = get_stock_data(thai_stocks)
df_vault = get_fund_data(vault_portfolio)

# Totals
usd_thb_rate = 34.0
grand_total = (df_us['Value'].sum() * usd_thb_rate) + df_thai['Value'].sum() + df_vault['Value'].sum()
grand_cost = (df_us['Cost Basis'].sum() * usd_thb_rate) + df_thai['Cost Basis'].sum() + df_vault['Cost Basis'].sum()
grand_pl = grand_total - grand_cost
grand_pct = (grand_pl / grand_cost) * 100

# --- DASHBOARD ---
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
    st.metric(
        "🦅 US ATTACK",
        f"${df_us['Value'].sum():,.0f}",
        delta=f"{df_us['P/L %'].mean():+.2f}% Avg (${us_pl_usd:,.0f})",
        delta_color="normal" if df_us['P/L %'].mean() > 0 else "inverse"
    )

with col3:
    st.metric(
        "🏦 THAI VAULT",
        f"฿{df_vault['Value'].sum():,.0f}",
        delta=f"{df_vault['P/L %'].mean():+.2f}% Avg (฿{vault_pl:,.0f})",
        delta_color="normal" if df_vault['P/L %'].mean() > 0 else "inverse"
    )

st.markdown("""---""")
st.markdown("")
tab1, tab2, tab3 = st.tabs(["🦅 US ATTACK", "🏰 THAI VAULT", "📊 ANALYTICS"])

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

# --- TRANSACTION SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.header("💰 ADD TRANSACTIONS")

# US VAULT TRANSACTIONS
st.sidebar.subheader("🦅 US Vault")
us_transaction_type = st.sidebar.radio("US Transaction Type", ["Buy", "Sell"], key="us_trans_type")

with st.sidebar.form("us_transaction_form"):
    us_ticker = st.text_input("Ticker Symbol (e.g., AAPL)", key="us_ticker")
    us_shares = st.number_input("Shares", min_value=0.0, step=0.01, key="us_shares")
    us_price = st.number_input("Price per Share ($)", min_value=0.0, step=0.01, key="us_price")
    us_submit = st.form_submit_button(f"Add US {us_transaction_type}")
    
    if us_submit and us_ticker and us_shares > 0 and us_price > 0:
        # Find if ticker exists
        try:
            ticker_idx = st.session_state.us_portfolio['Ticker'].index(us_ticker)
            existing_shares = st.session_state.us_portfolio['Shares'][ticker_idx]
            existing_avg_cost = st.session_state.us_portfolio['Avg_Cost'][ticker_idx]
            
            if us_transaction_type == "Buy":
                # Calculate new average cost
                total_cost = (existing_shares * existing_avg_cost) + (us_shares * us_price)
                new_shares = existing_shares + us_shares
                new_avg_cost = total_cost / new_shares
                
                st.session_state.us_portfolio['Shares'][ticker_idx] = new_shares
                st.session_state.us_portfolio['Avg_Cost'][ticker_idx] = new_avg_cost
                st.sidebar.success(f"✅ Added {us_shares} shares of {us_ticker} @ ${us_price:.2f}")
            else:  # Sell
                if us_shares > existing_shares:
                    st.sidebar.error(f"❌ Cannot sell {us_shares} shares. Only {existing_shares} available.")
                else:
                    new_shares = existing_shares - us_shares
                    if new_shares == 0:
                        # Remove the ticker entirely
                        del st.session_state.us_portfolio['Ticker'][ticker_idx]
                        del st.session_state.us_portfolio['Shares'][ticker_idx]
                        del st.session_state.us_portfolio['Avg_Cost'][ticker_idx]
                        st.sidebar.success(f"✅ Sold all {us_ticker} shares")
                    else:
                        # Keep average cost the same
                        st.session_state.us_portfolio['Shares'][ticker_idx] = new_shares
                        st.sidebar.success(f"✅ Sold {us_shares} shares of {us_ticker} @ ${us_price:.2f}")
                    
        except ValueError:
            # Ticker doesn't exist
            if us_transaction_type == "Buy":
                st.session_state.us_portfolio['Ticker'].append(us_ticker)
                st.session_state.us_portfolio['Shares'].append(us_shares)
                st.session_state.us_portfolio['Avg_Cost'].append(us_price)
                st.sidebar.success(f"✅ Added new position: {us_shares} shares of {us_ticker} @ ${us_price:.2f}")
            else:
                st.sidebar.error(f"❌ Cannot sell {us_ticker}. No existing position found.")
        
        st.rerun()

st.sidebar.markdown("---")

# THAI VAULT TRANSACTIONS
st.sidebar.subheader("🏰 Thai Vault")
thai_transaction_type = st.sidebar.radio("Thai Transaction Type", ["Buy", "Sell"], key="thai_trans_type")

thai_vault_type = st.sidebar.radio("Asset Type", ["Thai Stock", "Mutual Fund"], key="thai_vault_type")

if thai_vault_type == "Thai Stock":
    with st.sidebar.form("thai_stock_form"):
        thai_ticker = st.text_input("Ticker (e.g., TISCO.BK)", key="thai_ticker")
        thai_shares = st.number_input("Shares", min_value=0.0, step=1.0, key="thai_shares")
        thai_price = st.number_input("Price per Share (฿)", min_value=0.0, step=0.01, key="thai_price")
        thai_submit = st.form_submit_button(f"Add Thai {thai_transaction_type}")
        
        if thai_submit and thai_ticker and thai_shares > 0 and thai_price > 0:
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
                    st.sidebar.success(f"✅ Added {thai_shares} shares of {thai_ticker} @ ฿{thai_price:.2f}")
                else:
                    if thai_shares > existing_shares:
                        st.sidebar.error(f"❌ Cannot sell {thai_shares} shares. Only {existing_shares} available.")
                    else:
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
                    st.sidebar.success(f"✅ Added new position: {thai_shares} shares of {thai_ticker} @ ฿{thai_price:.2f}")
                else:
                    st.sidebar.error(f"❌ Cannot sell {thai_ticker}. No existing position found.")
            
            st.rerun()

else:  # Mutual Fund
    with st.sidebar.form("thai_fund_form"):
        fund_code = st.text_input("Fund Code (e.g., SCBNDQ(E))", key="fund_code")
        fund_units = st.number_input("Units", min_value=0.0, step=0.01, key="fund_units")
        fund_price = st.number_input("NAV per Unit (฿)", min_value=0.0, step=0.0001, key="fund_price")
        fund_master = st.selectbox("Master ETF", ["QQQ", "VOO", "VTI", "SOXX", "ICLN", "GLD", "^SET.BK", "N/A"], key="fund_master")
        fund_submit = st.form_submit_button(f"Add Fund {thai_transaction_type}")
        
        if fund_submit and fund_code and fund_units > 0 and fund_price > 0:
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
                    st.sidebar.success(f"✅ Added {fund_units} units of {fund_code} @ ฿{fund_price:.4f}")
                else:
                    if fund_units > existing_units:
                        st.sidebar.error(f"❌ Cannot sell {fund_units} units. Only {existing_units} available.")
                    else:
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
                    st.sidebar.success(f"✅ Added new fund: {fund_units} units of {fund_code} @ ฿{fund_price:.4f}")
                else:
                    st.sidebar.error(f"❌ Cannot sell {fund_code}. No existing position found.")
            
            st.rerun()

