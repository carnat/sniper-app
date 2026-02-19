# � SNIPER OS v1.2

A hybrid portfolio intelligence command center for tracking US stocks, Thai stocks, and Thai mutual funds with real-time data.

## Features

- 🦅 **US Stock Tracking** - Live prices via yfinance
- 🏰 **Thai Stock Tracking** - Live prices for Thai equities
- 🏦 **Thai Mutual Fund Tracking** - Real-time NAV from SEC Thailand API
- 📊 **Master ETF Correlation** - Track how funds perform vs their benchmark ETFs
- 💰 **Transaction Management** - Buy/Sell transactions with automatic cost basis calculation

## Setup

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Configure Your Portfolio (Private Data)

Your holdings are stored in `.streamlit/secrets.toml` which is **NOT tracked by Git**.

1. Edit `.streamlit/secrets.toml`
2. Add your holdings following the examples in the file
3. Save and restart the app

**Example:**
```toml
[us_portfolio]
Ticker = ["AAPL", "GOOGL"]
Shares = [10.0, 5.0]
Avg_Cost = [150.25, 2800.50]

[[vault_portfolio]]
Code = "SCBNDQ(E)"
Units = 1000
Cost = 13.50
Master = "QQQ"
```

### 3. Run the App

```bash
streamlit run streamlit_app.py
```

## Security

- ✅ Portfolio data is stored in `.streamlit/secrets.toml` (gitignored)
- ✅ Never commit your holdings to Git
- ✅ Safe to push code to GitHub - your data stays private

## Deployment on Streamlit Cloud

When deploying to Streamlit Cloud:
1. Go to your app settings
2. Navigate to "Secrets" section
3. Copy the contents of your local `.streamlit/secrets.toml`
4. Paste into the Streamlit Cloud secrets manager
