# 🎯 SNIPER OS

**HYBRID INTEL COMMAND CENTER** - A comprehensive portfolio intelligence dashboard for tracking US stocks, Thai equities, and Thai mutual funds with real-time market data and performance analytics.

Current app version is stored in [VERSION](VERSION).

## 🚀 Features

### 📊 Portfolio Overview Dashboard

- **Net Worth (THB)** - Combined portfolio value with P/L tracking across all assets
- **US Attack Performance** - US equity portfolio metrics with average returns
- **Thai Vault Performance** - Thai holdings metrics with average returns
- Real-time value updates and P/L calculations

### 🦅 US Equities (`US ATTACK`)

- Track US stocks with live prices from yfinance
- Display shares, average cost, live price, and P/L metrics
- Automatic cost basis calculations
- Color-coded P/L % (green for gains, red for losses)

### 🏰 Thai Assets (`THAI VAULT`)

- **Thai Equities** - Real-time price tracking for Thai-listed stocks (.BK tickers)
- **Mutual Funds** - Real-time NAV (Net Asset Value) from SEC Thailand API
  - Fund daily gains tracking
  - Master ETF correlation analysis
  - Performance comparison vs benchmark ETFs (QQQ, VOO, SOXX, VTI, GLD, etc.)
  - Master vs Fund % - Shows daily correlation difference between fund and master ETF

### 📈 Analytics Tab

- Performance charts for US equities
- Performance charts for mutual funds
- Visual P/L % comparisons across holdings

### � News Watchtower (NEW!)

- **Real-Time News Feed** - Latest articles for each holding from NewsAPI
- **Price Alerts** - Automatic alerts when prices move ±5% (customizable)
- **Smart Filtering** - View news by ticker with direct article links
- **Alert Dashboard** - Visual indicators for significant price movements
- See [NEWS_SETUP.md](NEWS_SETUP.md) for configuration

### �💰 Transaction Management (Sidebar)

- **Buy/Sell Transactions** for US stocks, Thai stocks, and Thai mutual funds
- **Automatic Cost Basis Updates** - Calculates weighted average cost
- Add new positions or modify existing ones
- Sell specific quantities while maintaining cost tracking

### 📡 Real-Time Data Sources

- ✅ **US/Thai Stocks**: yfinance (market hours data)
- ✅ **Thai Mutual Funds**: Official SEC Thailand API (api.sec.or.th)
- ✅ **Master ETF Tracking**: Real-time market data from yfinance
- ✅ **Live NAV**: Direct integration with Thai fund registry

## 📋 Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Configure Your Portfolio (Private Data)

Your holdings are stored in `.streamlit/secrets.toml` which is **NOT tracked by Git**.

**Edit `.streamlit/secrets.toml`:**

```toml
# US Portfolio
[us_portfolio]
Ticker = ["AAPL", "GOOGL", "MSFT"]
Shares = [10.0, 5.0, 15.0]
Avg_Cost = [150.25, 2800.50, 380.00]

# Thai Stocks
[thai_stocks]
Ticker = ["TISCO.BK", "ADVANC.BK"]
Shares = [100, 50]
Avg_Cost = [95.50, 280.00]

# Mutual Funds (add a [[vault_portfolio]] section for each fund)
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

# Optional: News & Alerts (requires free NewsAPI key)
[news_alerts]
newsapi_key = "your_free_key_from_newsapi.org"
price_alert_threshold = 5
enable_price_alerts = true
enable_news_feed = true
```

**Get NewsAPI Key:** Sign up free at [newsapi.org](https://newsapi.org/) (100 requests/day)

### 3. Run the App

```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` in your browser

## 🔒 Security

Your portfolio data stays **completely private**:

- ✅ **Portfolio data** stored in `.streamlit/secrets.toml` (gitignored)
- ✅ **Never committed to Git** - safe to push code to GitHub
- ✅ **Local storage** - data remains on your machine
- ✅ **No telemetry** - your holdings are never sent anywhere
- ✅ **Git history cleaned** - old commits with sensitive data removed

### Portfolio Data Format

**US Portfolio:**

- Ticker (string array) - US stock symbols
- Shares (float array) - Number of shares held
- Avg_Cost (float array) - Average cost per share (USD)

**Thai Stocks:**

- Ticker (string array) - Thai stock symbols ending with `.BK`
- Shares (float array) - Number of shares held
- Avg_Cost (float array) - Average cost per share (THB)

**Vault Portfolio (Mutual Funds):**

- Code (string) - Fund identifier (e.g., "SCBNDQ(E)")
- Units (float) - Number of fund units
- Cost (float) - NAV cost per unit (THB)
- Master (string) - Benchmark ETF for correlation tracking

## 🌐 Deployment on Streamlit Cloud

To deploy your app securely on Streamlit Cloud:

1. Push code to GitHub (secrets.toml is gitignored, so your data won't be pushed)
2. Connect your GitHub repo to Streamlit Cloud
3. Go to app settings → "Secrets"
4. Copy contents of your local `.streamlit/secrets.toml`
5. Paste into the Streamlit Cloud secrets manager
6. Your app will load holdings from Streamlit Cloud secrets

## 📊 Dashboard Metrics Explained

- **Net Worth (THB)** - Total portfolio value converted to Thai Baht (USD holdings × 34 THB/USD)
- **P/L** - Unrealized profit/loss (Current Value - Cost Basis)
- **P/L %** - Percentage return on investment
- **Fund Day Gain %** - Day-over-day NAV change for mutual funds
- **Master Day Gain %** - Day-over-day performance of benchmark ETF
- **Master vs Fund %** - Difference between fund and master performance (alpha tracking)

## 🛠️ Technical Stack

- **Framework**: Streamlit
- **Data**: yfinance, SEC Thailand API
- **Storage**: TOML secrets file
- **Language**: Python 3.8+

## 🔢 Software Versioning

This project uses **Semantic Versioning** (`MAJOR.MINOR.PATCH`):

- `MAJOR` - breaking changes
- `MINOR` - backward-compatible features
- `PATCH` - backward-compatible fixes

### Single Source of Truth

- App version is read from [VERSION](VERSION)
- Release notes are tracked in [CHANGELOG.md](CHANGELOG.md)

### Bump Version

```bash
python bump_version.py patch
# or: python bump_version.py minor
# or: python bump_version.py major
```

After bumping:

1. Update [CHANGELOG.md](CHANGELOG.md)
2. Commit version + changelog together
3. Tag release in Git (optional)

## 📝 Notes

- Prices update in real-time during market hours
- Thai market data available during Thai market hours
- Average cost basis automatically calculated on buy transactions
- All prices and costs denominated in their respective currencies (USD for US, THB for Thai)

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [SEC Thailand API](https://api.sec.or.th/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

---

**Disclaimer**: This is a personal portfolio tracking tool. Not financial advice. Always do your own research before investing.
