# 🎯 SNIPER OS

[![CI](https://github.com/carnat/sniper-app/actions/workflows/ci.yml/badge.svg)](https://github.com/carnat/sniper-app/actions/workflows/ci.yml)

Streamlit portfolio dashboard for US stocks, Thai equities, and Thai mutual funds.

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Configure `.streamlit/secrets.toml`

Minimum required structure:

```toml
[us_portfolio]
Ticker = ["AAPL"]
Shares = [1.0]
Avg_Cost = [150.0]

[thai_stocks]
Ticker = ["ADVANC.BK"]
Shares = [1.0]
Avg_Cost = [200.0]

[[vault_portfolio]]
Code = "SCBNDQ(E)"
Units = 100.0
Cost = 10.0
Master = "QQQ"
```

News Watchtower setup: [NEWS_SETUP.md](NEWS_SETUP.md)

### 3) Run

```bash
streamlit run streamlit_app.py
```

## Notes

- `.streamlit/secrets.toml` is local and gitignored.
- Secret scanner: `python scripts/secret_scan.py`
