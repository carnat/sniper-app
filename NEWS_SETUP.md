# 📰 NEWS WATCHTOWER SETUP GUIDE

## Quick Start

### 1. Get Your FREE Finnhub Key

1. Visit https://finnhub.io/
2. Click "Get API Key" (no credit card required)
3. Register with your email
4. Copy your API key

### 2. Configure Secrets

Edit `.streamlit/secrets.toml` and add:

```toml
[news_alerts]
finnhub_api_key = "your_finnhub_api_key_here"
price_alert_threshold = 5  # Alert if price moves ±5%
enable_price_alerts = true
enable_news_feed = true
enable_earnings_alerts = true
```

### 3. Restart Streamlit

```bash
streamlit run streamlit_app.py
```

## Features

### 📰 News Feed
- View latest news for any holding
- Source: Finnhub company-news
- Direct links to full articles
- Automatic filtering by ticker

### 🚨 Price Alerts
- Automatic alerts when holdings move ±5% (or custom threshold)
- Shows current price vs cost basis
- Visual indicators for up/down movements
- Works for both US and Thai stocks

### 📊 Smart Filtering
- Finnhub free tier available for `company-news`
- Results cached for 1 hour to save API calls
- Automatic retry logic

## Troubleshooting

**"Finnhub key not configured"**
- Make sure you added the key to `.streamlit/secrets.toml`
- Restart the Streamlit app after adding the key

**No news appearing:**
- Some symbols have lower coverage on certain days
- Try different holdings - some tickers have more coverage
- Check your API key is valid on Finnhub dashboard

**Rate limit exceeded:**
- Wait and retry later or upgrade plan
- Results are cached for 1 hour to minimize requests

## Advanced Configuration

Adjust thresholds in secrets.toml:

```toml
[news_alerts]
price_alert_threshold = 10  # More conservative (±10%)
# or
price_alert_threshold = 3   # More sensitive (±3%)
```

---

**Next Phase:** Technical indicators, earnings calendar, and sentiment analysis coming soon! 🚀
