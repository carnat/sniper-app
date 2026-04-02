# scripts/ — CLAUDE.md

Price pipeline only. This directory fetches market data and writes to `data/prices.json`.
Output is read-only JSON. No portfolio decisions. No execution logic.

## Files
- `fetch_prices.py` — reads tickers from config, calls yfinance, writes `data/prices.json`
- `encrypt_private.py` — encrypts `command_center/private.json` → `command_center/private.enc.json` (AES-256-GCM, PBKDF2)
- `claude_guardrail.py` — pre-tool hook: blocks dangerous bash commands before execution
- `secret_scan.py` — scans repo for accidentally committed secrets before push

## Ticker Map (doctrine_ops v1.15.1 §0.4)
- **Arsenal (10):** VRT, ASTS, VST, MU, APH, ANET, TSM, ONDS, FN, COHR
- **Watchtower Core (12):** TSEM, BWXT, MOD, NBIS, FORM, ENTG, ONTO, LITE, QRVO, PLTR, KTOS, SKYT
- **Watchtower Satellite (3):** RKLB, SATL, PL
- **Sector ETFs:** SOXX, ITA, XLU
- **Benchmarks:** VOO, QQQ, EWY, GLD, SPY
- **Macro:** ^VIX, THB=X

## prices.json Schema (per-ticker)
`price`, `change` (daily%), `change5d` (5d%), `adv` (20d avg vol), `volume`, `high52w`, `low52w`, `sma50`, `sma200`, `news` (array of `{title, url, date}`, max 3)

## prices.json Metadata Keys
`last_updated`, `fx_rate` (THB/USD), `vix`, `spy_above_200dma` (bool), `regime` (GREEN/YELLOW/ORANGE/RED)

## Regime Bands (doctrine_core v1.12.1)
- GREEN: VIX < 22
- YELLOW: 22 ≤ VIX < 25
- ORANGE: 25 ≤ VIX ≤ 30 (FREEZE — deployment blocked)
- RED: VIX > 30 (HARD FREEZE)

## Rate Limiting
Use `ratelimit` library for news fetch calls. Batch OHLCV download via `yf.download()`.

## Constraints
Do not add any file here that:
- Modifies `.streamlit/` or any file listed in `.gitignore`
- Triggers external APIs beyond yfinance and the `cryptography` stdlib
- Contains portfolio execution logic of any kind
