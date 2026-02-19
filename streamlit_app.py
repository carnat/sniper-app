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
