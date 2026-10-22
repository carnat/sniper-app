# scripts/ — CLAUDE.md

Price pipeline, encryption, and portfolio import. Outputs are read-only JSON. No portfolio decisions. No execution logic.

## Files
- `fetch_prices.py` — reads tickers from config, calls yfinance, writes `data/prices.json`
- `encrypt_private.py` — encrypts `command_center/private.json` → `command_center/private.enc.json` (AES-256-GCM, PBKDF2)
- `import_portfolio.py` — imports `data/portfolio_config.json` (gitignored) → `data/portfolio.json` (gitignored). Validates against doctrine rules. Optional `--encrypt` flag.
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
`price`, `change` (daily%), `change5d` (5d%), `adv` (20d avg vol), `volume`, `high52w`, `low52w`, `sma50`, `sma200`, `market_time`, `market_date` (YYYY-MM-DD of the close), `stale` (bool), `sessions_behind` (int), `data_quality` (list of codes), `news` (array of `{title, url, date}`, max 3)

## prices.json Metadata Keys
`last_updated`, `fx_rate` (THB/USD), `vix`, `spy_above_200dma` (bool), `regime` (GREEN/YELLOW/ORANGE/RED), `reference_market_date` (newest session in the feed), `stale_tickers` (list), `missing_tickers` (configured but not in this refresh)

## Per-ticker data quality (not doctrine)
`last_updated` is when the pipeline ran. It says nothing about each quote's age: one
symbol can lag inside an otherwise fresh refresh (QRVO stuck on its 2026-10-05 close in
the 2026-10-22 refresh). `annotate_freshness()` handles this:
- Reference session = newest `market_date` among `PRICE_TICKERS` (macro excluded: other calendars).
- `sessions_behind` = weekdays from the ticker's session to the reference. `stale` = `sessions_behind >= 1`.
- `volume` is read from the same session as the close. NaN or 0 on that session is a
  missing observation: `null` + `volume_unavailable`, never `0`. ADV averages only real (>0) prior sessions.
- `sma50`/`sma200` stay `null` with fewer than 50/200 closes: `sma50_unavailable`/`sma200_unavailable`.
- `data_quality` codes: `stale_quote`, `volume_unavailable`, `sma50_unavailable`, `sma200_unavailable`.
- ADV section: a stale ticker keeps its ratio but `gate_met` is `false` and `stale: true`.
  A ticker without real volume has no ADV entry.
- Stale quotes are kept (not dropped) and flagged; the Command Center badges them.

Doctrine reconciliation: `docs/doctrine/doctrine_ops.md` RULE 7 says price data older than
1 business day is marked STALE in gate check outputs. Before this change the code only
checked the feed run time (Command Center banner, trading-day aware). Now each ticker is
also checked against the feed's own newest session. The pipeline does not yet compare
the reference session against the run date, and no gate output prints a
"STALE DATA" caveat. Doctrine text is unchanged.

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
- Triggers external APIs beyond yfinance and `cryptography`
- Contains portfolio execution logic of any kind
