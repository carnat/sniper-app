# scripts/ — CLAUDE.md

Price pipeline only. This directory fetches market data and writes to `data/prices.json`.
Output is read-only JSON. No portfolio decisions. No execution logic.

## Files
- `fetch_prices.py` — reads tickers from config, calls yfinance, writes `data/prices.json`
- `encrypt_private.py` — encrypts `command_center/private.json` → `command_center/private.enc.json` (AES-256-GCM, PBKDF2)
- `claude_guardrail.py` — pre-tool hook: blocks dangerous bash commands before execution
- `secret_scan.py` — scans repo for accidentally committed secrets before push

## Constraints
Do not add any file here that:
- Modifies `.streamlit/` or any file listed in `.gitignore`
- Triggers external APIs beyond yfinance and the `cryptography` stdlib
- Contains portfolio execution logic of any kind
