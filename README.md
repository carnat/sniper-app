# 🎯 SNIPER OS

SNIPER OS is a Streamlit portfolio dashboard for tracking US stocks, Thai equities, and Thai mutual funds in one place.

It combines live market pricing, transaction tracking, portfolio analytics, risk snapshots, and scenario tools in a single app.

Current release version: [VERSION](VERSION)

## Why SNIPER OS

- Multi-market tracking for US + Thailand holdings
- Realized and unrealized P/L visibility
- Portfolio analytics (attribution, risk panel, rebalancing what-if)
- News and alert workflow tied to portfolio symbols
- Local-first storage with secrets-based private configuration

## Feature Highlights

### Portfolio Overview

- Net worth and P/L summary in THB
- Separate US and Thai portfolio performance blocks
- "Today Brief" indicators for quick daily context

### Holdings & Pricing

- US/Thai equities via yfinance
- Thai mutual fund NAV via SEC Thailand API
- Fund-to-master comparison metrics (e.g., QQQ, VOO)

### Transactions & Lots

- Sidebar trade entry for US stocks, Thai stocks, and funds
- Buy/Sell + cash events (Dividend/Fee) + stock splits
- Lot methods per asset class (FIFO/LIFO/AVERAGE)
- Realized P/L history and correction/reversal workflow

### Analytics

- P/L attribution by asset class and instrument
- Risk Panel v1 (concentration, stress assumptions, exposure mix)
- What-if rebalancing calculator
- Signal scoring layer
- Snapshot history and scenario backtesting/compare

### News & Alerts

- News feed by holding (NewsAPI)
- Price alert monitoring based on configured thresholds

See [NEWS_SETUP.md](NEWS_SETUP.md) for NewsAPI setup.

## Screenshots

Add screenshots to [docs/screenshots](docs/screenshots) and update the image links below.

![Dashboard Overview](docs/screenshots/dashboard-overview.png)
![Analytics Risk Panel](docs/screenshots/analytics-risk-panel.png)
![News Watchtower](docs/screenshots/news-watchtower.png)

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Add private portfolio config

Create or edit `.streamlit/secrets.toml`:

```toml
[us_portfolio]
Ticker = ["AAPL", "GOOGL", "MSFT"]
Shares = [10.0, 5.0, 15.0]
Avg_Cost = [150.25, 2800.50, 380.00]

[thai_stocks]
Ticker = ["TISCO.BK", "ADVANC.BK"]
Shares = [100, 50]
Avg_Cost = [95.50, 280.00]

[[vault_portfolio]]
Code = "SCBNDQ(E)"
Units = 1000
Cost = 13.50
Master = "QQQ"

[[vault_portfolio]]
Code = "SCBS&P500FUND(E)"
Units = 500
Cost = 38.00
Master = "VOO"

[news_alerts]
newsapi_key = "your_free_key_from_newsapi.org"
price_alert_threshold = 5
enable_price_alerts = true
enable_news_feed = true

[sec_api]
primary_key = "your_sec_primary_key"
secondary_key = "your_sec_secondary_key"
```

Get a free key at [newsapi.org](https://newsapi.org/).

### 3) Run the app

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Privacy & Data Storage

- Portfolio secrets: `.streamlit/secrets.toml` (gitignored)
- Local runtime state: `.streamlit/` files (transactions, lots DB, snapshots, alerts)
- No dedicated portfolio telemetry pipeline in this repo

## Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Connect repo in Streamlit Cloud
3. Open app settings → Secrets
4. Paste local `.streamlit/secrets.toml` contents
5. Deploy and verify portfolio loads

## Software Versioning

This project uses Semantic Versioning (`MAJOR.MINOR.PATCH`).

- Source of truth: [VERSION](VERSION)
- Release notes: [CHANGELOG.md](CHANGELOG.md)

### Bump version

```bash
python bump_version.py patch
# or: python bump_version.py minor
# or: python bump_version.py major
```

### Release flow (example)

```bash
python bump_version.py patch

# update CHANGELOG.md
git add VERSION CHANGELOG.md
git commit -m "Release vX.Y.Z"

git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main
git push origin vX.Y.Z
```

## Tech Stack

- Streamlit
- pandas
- yfinance
- SEC Thailand API

## Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [SEC Thailand API](https://api.sec.or.th/)
- [yfinance](https://github.com/ranaroussi/yfinance)

---

**Disclaimer:** This tool is for portfolio tracking and research workflows. It is not financial advice.
