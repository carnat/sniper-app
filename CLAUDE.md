# SNIPER APP — CLAUDE.md

## What this project is
Portfolio OS for the Sniper Doctrine rules-based investment system. Two tools in one repo:
1. `streamlit_app.py` — full portfolio management (transactions, analytics, tax lots, options)
2. `command_center/index.html` — static offline dashboard, daily situational awareness

## Stack
Python 3.11 / Streamlit / yfinance / pandas / SQLite / ratelimit
HTML + vanilla JS (no framework) / Chart.js / JetBrains Mono + Rajdhani fonts

## Commands
```
streamlit run streamlit_app.py          # run Streamlit app
python scripts/fetch_prices.py          # generate data/prices.json
pytest tests/                           # run test suite
python scripts/secret_scan.py          # check for secrets before commit
python scripts/encrypt_private.py      # encrypt private.json → private.enc.json
```

## Architecture decisions
- Portfolio state: `.streamlit/secrets.toml` (gitignored — NEVER commit, NEVER generate)
- Price data flow: GitHub Action → `scripts/fetch_prices.py` → `data/prices.json` → `command_center/index.html`
- Private data: `command_center/private.json` (gitignored, local dev) or `command_center/private.enc.json` (AES-GCM encrypted, safe to commit) — decrypted client-side via PIN
- Streamlit: server-required, deep analytics, transaction entry, tax lots
- Command Center HTML: offline-capable, no server, daily gate check
- Doctrine rules: live in `.sniper-plugin/skills/` — do NOT duplicate in this file
- Subfolder `CLAUDE.md` files narrow context per domain — respect them

## Critical constraint — COMMANDER-IN-LOOP
No execution automation. No auto-trade. No auto-order. No broker API. Alert systems are
informational only. This rule overrides all other considerations. If a feature would remove
Commander from the decision chain, do not build it.

## What NOT to do
- Never modify `.streamlit/secrets.toml` or generate it
- Never add broker API integration of any kind
- Never use `/gsd auto` mode — step mode only
- Never touch transaction/lot database logic without explicit Commander instruction
- Never duplicate doctrine rules outside `.sniper-plugin/skills/`
- Never add `node_modules`, `package.json`, `webpack`, or build tooling to `command_center/`

## Session order
| Session | Task | Key output | Status |
|---------|------|------------|--------|
| 0 | CLAUDE.md hierarchy + hooks + guardrail + plugin scaffold | Root + subfolders CLAUDE.md, `.claude/settings.json`, `claude_guardrail.py`, `.sniper-plugin/` skeleton | ✅ Complete |
| 1 | Price pipeline + wire Command Center | `fetch_prices.py`, `update_prices.yml`, `index.html` reads `data/prices.json` | ✅ Complete |
| 1b | Privacy layer + PIN encryption | `private.enc.json`, `encrypt_private.py`, PIN modal in `index.html` | ✅ Complete |
| 2 | Streamlit restyle + plugin skills population | CSS block replaced, `skills/` files populated | ✅ Complete |
| 3 | CMD-2 Watchtower panel + ADV/news pipeline + price pipeline upgrade | 3rd tab in `index.html`; `fetch_prices.py` doctrine v1.15.1 tickers + enriched data (SMA, 52w, regime) | ✅ Complete |
| 4 | CMD-3 Live Backtester panel | 4th tab in `index.html` — DCA simulation, Matrix allocation vs equal-weight | ✅ Complete |
| 4b | CMD-4 War Room — Agent-Based Pixel Art | 5th tab in `index.html` — Canvas 2D game engine, 28 pixel agents, regime effects, speech bubbles | ✅ Complete |
| 5 | CMD-5 Triad Council — Doctrine Audit Overlay | 3 council agents (DOCTRINE/GATE/WATCHER) in War Room canvas; audit engine with rotating findings, consensus detection, click tooltips | ✅ Complete |
| 6+ | P7 alert pipeline | Post-Level 1 (฿800K trigger) | 🔜 Not started |

## Reference
Full session handoff: `docs/sniper_claude_code_handoff.md`
