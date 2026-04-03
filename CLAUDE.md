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
python scripts/import_portfolio.py --template    # generate blank portfolio config
python scripts/import_portfolio.py               # import portfolio from config
python scripts/import_portfolio.py --encrypt     # import + encrypt
```

## Architecture decisions
- Portfolio state: `.streamlit/secrets.toml` (gitignored — NEVER commit, NEVER generate)
- Price data flow: GitHub Action → `scripts/fetch_prices.py` → `data/prices.json` → `command_center/index.html`
- Private data: `command_center/private.json` (gitignored, local dev) or `command_center/private.enc.json` (AES-GCM encrypted, safe to commit) — decrypted client-side via PIN
- Streamlit: server-required, deep analytics, transaction entry, tax lots
- Command Center HTML: offline-capable, no server, daily gate check
- Doctrine rules: authoritative full text in `docs/doctrine/`, compressed summaries in `.sniper-plugin/skills/`
- Portfolio import: `scripts/import_portfolio.py` reads `data/portfolio_config.json` (gitignored) → `data/portfolio.json` (gitignored)
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
- Never duplicate doctrine rules outside `.sniper-plugin/skills/` and `docs/doctrine/`
- Never add `node_modules`, `package.json`, `webpack`, or build tooling to `command_center/`
- Never hardcode actual portfolio values (shares, cost basis, ammo) in committed files — use gitignored data files
- Never create phantom/estimated portfolio positions — 0 shares = 0 shares (CA-3)
- Never confuse cost-basis P&L for thesis health — they are separate concerns

## Doctrine versions (authoritative source: `docs/doctrine/`)
| File | Version | Status |
|------|---------|--------|
| `doctrine_core.md` | v1.12.1 | Current |
| `doctrine_ops.md` | v1.15.1 | Current |
| `doctrine_council.md` | v1.11.0 | Current |
| `doctrine_pe.md` | v1.6.0 | Current |
| `doctrine_core_matrix.md` | v1.0.0-BETA.1 | Current |
| `doctrine_core_tripwires.md` | v1.0.0-BETA.1 | Current |

## Ticker map (doctrine_ops v1.15.1 §0.4)
- **Arsenal (10):** VRT, ASTS, VST, MU, APH, ANET, TSM, ONDS, FN, COHR
- **Watchtower Core (12):** TSEM, BWXT, MOD, NBIS, FORM, ENTG, ONTO, LITE, QRVO, PLTR, KTOS, SKYT
- **Watchtower Satellite (3):** RKLB, SATL, PL

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
| 7 | Doctrine integration + portfolio import | `docs/doctrine/`, `scripts/import_portfolio.py`, portfolio config schema | ✅ Complete |
| 8 | sniper/ package extraction | Business logic → 11 modules, `streamlit_app.py` 5681→3827 lines | ✅ Complete |
| 9 | CMD-6→11 UI/UX enhancements | PRI gauge, signal convergence, sparklines, ticker tape, news badges, keyboard shortcuts, auto-refresh, mobile tab bar, panel manager | ✅ Complete |
| 10 | Handoff doc + CLAUDE.md overhaul | Docs aligned to current state, stale data removed, doctrine gap analysis | ✅ Complete |

## Next priorities (doctrine gap alignment)
- DCA Matrix live scoring engine (F1-F4 computed, not hardcoded)
- Drawdown Freeze enforcement (HWM -20% blocks DCA, Amber Zone)
- Victory Protocol tracking (2x/3x triggers, Trailing Shield)
- Concentration limit enforcement (25% single, 15% satellite, 40% sector)
- Alpha Filter validation panel (China 20/20, ADV gate)
- Enhanced Triad Council audits (full gate decision tree)
- Bear Restructure logic (RED >10 sessions)
- Dead Hand Clause (120-day /review trigger)

## Reference
Authoritative doctrine: `docs/doctrine/` (version-controlled, 6 files)
Handoff doc: `docs/sniper_claude_code_handoff.md` (⚠️ deprecated — superseded by doctrine folder + CLAUDE.md hierarchy)
