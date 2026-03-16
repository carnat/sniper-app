# SNIPER TOOLS — CLAUDE CODE HANDOFF

**Date:** Mar 16, 2026
**From:** Sniper Spinoff Project (claude.ai)
**To:** Claude Code — sniper-tools repo

---

## WHAT YOU ARE BUILDING

A personal investment command center for the Sniper Doctrine system. Two tools in one repo:
1. `command_center/index.html` — static offline dashboard, doctrine-aware gate status, daily situational awareness
2. `streamlit_app.py` — full portfolio OS (transactions, analytics, tax lots, options greeks) — restyled to match Command Center aesthetic

**Non-negotiable constraint:** No feature may auto-execute trades, auto-send orders, or remove Commander from the decision chain. Alert systems are informational only.

---

## REPO STRUCTURE (existing: carnat/sniper-app)

```
carnat/sniper-app/
├── CLAUDE.md                     ← root context + constraints
├── streamlit_app.py              ← EXISTS — restyle CSS only, no logic changes
├── requirements.txt              ← EXISTS (streamlit, yfinance, pandas, ratelimit, requests, cryptography)
├── scripts/
│   ├── CLAUDE.md                 ← price pipeline only, read-only output
│   ├── secret_scan.py            ← EXISTS
│   ├── fetch_prices.py           ← writes data/prices.json
│   ├── encrypt_private.py        ← encrypts private.json → private.enc.json (AES-GCM)
│   └── claude_guardrail.py       ← pre-tool hook safety check
├── data/
│   └── prices.json               ← GitHub Action output
├── .github/workflows/
│   ├── ci.yml                    ← EXISTS (keep)
│   ├── update_prices.yml         ← daily yfinance fetch
│   └── pages.yml                 ← GitHub Pages deploy
├── .claude/
│   └── settings.json             ← hooks config (Commander-in-loop enforcement)
├── command_center/
│   ├── CLAUDE.md                 ← static HTML only, no Python, no server
│   ├── index.html                ← Command Center v2 (PIN encryption, data/prices.json)
│   ├── private.json              ← GITIGNORED — local dev plaintext private data
│   └── private.enc.json          ← AES-GCM encrypted — safe to commit, PIN-decrypted in browser
├── .sniper-plugin/               ← Session 2: Sniper plugin
│   ├── .claude-plugin/
│   │   └── plugin.json           ← manifest
│   ├── .mcp.json                 ← Yahoo Finance MCP wired here
│   ├── commands/
│   │   ├── scorecard.md          ← /sniper:scorecard
│   │   ├── thesis.md             ← /sniper:thesis [TICKER]
│   │   ├── brief.md              ← /sniper:brief
│   │   └── startup.md            ← /sniper:startup
│   └── skills/
│       ├── doctrine-core.md      ← Alpha Filters, gate rules, DC rules — auto-activates
│       ├── portfolio-state.md    ← positions, ammo, Watchtower — auto-activates
│       └── vault-rules.md        ← vault switch triggers, E-class logic — auto-activates
├── index.html                    ← root redirect → ./command_center/
└── .streamlit/
    └── secrets.toml              ← EXISTS (gitignored — local only, NEVER commit)
```

---

## STATE DATA (hardcoded JS defaults in command_center/index.html)

```javascript
// Arsenal Core positions
{ ticker: 'VRT',  shares: 0.148355, cost: 235.92, thesis: 'Mar 16, 2026' }
{ ticker: 'VST',  shares: 12,       cost: 158.32, thesis: 'Mar 10, 2026' }
{ ticker: 'MU',   shares: 3.062527, cost: 294.01, thesis: 'Mar 15, 2026' }
{ ticker: 'APH',  shares: 8,        cost: 136.95, thesis: 'Mar 16, 2026' }
{ ticker: 'ANET', shares: 3,        cost: 122.12, thesis: 'Mar 10, 2026' }
{ ticker: 'TSM',  shares: 0.08434,  cost: 296.42, thesis: 'Mar 16, 2026' }
{ ticker: 'FN',   shares: 0,        cost: null,   thesis: 'Mar 16, 2026' }

// Arsenal Satellites
{ ticker: 'ASTS', shares: 9.3,  cost: 74.50 }
{ ticker: 'ONDS', shares: 1,    cost: 9.05  }

// Cached prices (Mar 13, 2026 — stale fallback)
VRT: 258.88, ASTS: 86.34, VST: 158.95, MU: 426.13
APH: 133.92, ANET: 133.57, TSM: 338.31, ONDS: 10.16
FN: 502.14, BWXT: 194.13, NBIS: 112.95
^VIX: 27.19, THB=X: 32.42

// Watchtower (5/6 slots)
PLTR  — Core      — Remove Jun 12
TSEM  — Core      — ⚠ ISRAEL WATCH — Remove Jun 10
NBIS  — Core      — Option C active — ⚠ EXPIRY MAR 26 — Remove Jun 10
BWXT  — Core      — ⚠ Blackout ~Apr 14 — Remove Sep 15
KTOS  — Satellite — ⚠ Blackout ~Apr 17 — Remove Sep 15

// YTD benchmark reference prices (Jan 1, 2026 close)
SOXX: 313.69, VOO: 628.30, QQQ: 613.12, EWY: 102.22
```

---

## PRIVATE DATA (PIN-ENCRYPTED)

Public page shows: prices, P&L %, gate status, thesis dates.
Hidden by default: shares count, ₿ values, vault THB totals.

**Local dev:** `command_center/private.json` (gitignored) — loaded without PIN
**Production:** `command_center/private.enc.json` (committed) — PIN modal on load

To re-encrypt after updating `private.json`:
```bash
python scripts/encrypt_private.py
git add command_center/private.enc.json
git commit -m "chore: update encrypted private data"
git push
```

---

## data/prices.json SCHEMA

```json
{
  "updated": "2026-03-16T10:00:00Z",
  "updated_bkk": "2026-03-16 17:00 GMT+7",
  "prices": {
    "VRT":   { "price": 258.88, "prev_close": 255.00, "change_pct": 1.52 },
    "^VIX":  { "price": 27.19 },
    "THB=X": { "price": 32.42 },
    "SOXX":  { "price": 331.32, "ytd_pct": 5.6 }
  },
  "ytd_ref": { "SOXX": 313.69, "VOO": 628.30, "QQQ": 613.12, "EWY": 102.22 },
  "gate": {
    "vix_freeze": true,
    "vix": 27.19,
    "thb_zone": "B",
    "thb": 32.42
  }
}
```

---

## DOCTRINE RULES (gate logic reflected in UI)

**VIX Gate:**
- VIX < 25 → CLEAR (green) → deployment allowed
- VIX ≥ 25 → FREEZE (amber, pulsing) → all deployment blocked
- ASTS SO-1 exempt from VIX gate (DC-05 pre-placed order carve-out)

**THB/USD FX Zones:**
- < 32.00 → Zone A (cyan) → deploy full ammo immediately
- 32.00–36.00 → Zone B (green) → normal deployment
- > 36.00 → Zone C (amber) → split 50/50 deploy/hold

**DC-06 Neglect Clocks:**
- Day 0 = Mar 13, 2026 (ASTS and ONDS — NOT SET positions only)
- Day 30 → ⚠ NEGLECT flag
- Day 60 → forced /council trim (red, blinking)

**BETA Alert:** 4 files pending upload to main doctrine Project Knowledge:
- `doctrine_core_v1_4_5_BETA_1.md`
- `doctrine_ops_v1_5_4_BETA_1.md`
- `portfolio_state_v1_0_17_BETA_1.md`
- `doctrine_pe_v1_2_1_BETA_1.md`

---

## SESSION ORDER

| Session | Task | Key output | Status |
|---------|------|------------|--------|
| 0 | CLAUDE.md hierarchy + hooks + guardrail + plugin scaffold | Root + subfolders CLAUDE.md, `.claude/settings.json`, `claude_guardrail.py`, `.sniper-plugin/` skeleton | ✓ DONE |
| 1 | Price pipeline + wire Command Center | `fetch_prices.py`, `update_prices.yml`, `index.html` reads `data/prices.json` | ✓ DONE |
| 1b | Privacy layer + PIN encryption | `private.enc.json`, `encrypt_private.py`, PIN modal | ✓ DONE |
| 2 | Streamlit restyle + plugin skills population | CSS block replaced, `skills/` files populated with doctrine content | NEXT |
| 3 | CMD-2 Watchtower panel | 3rd tab in `index.html` | — |
| 4 | CMD-3 Live Backtester panel | 4th tab | — |
| 5+ | P7 alert pipeline | Post-Level 1 (฿800K trigger) | — |

---

## SESSION 2 — STREAMLIT RESTYLE (CSS only, no logic changes)

Replace CSS injection block at line 33 of `streamlit_app.py`. See full spec in original handoff.

**Color palette:**
```
--bg-0: #070809   --bg-1: #0d1014   --bg-2: #131820   --bg-3: #1a2030
--green: #00d084  --amber: #ffb800  --red: #ff4055    --blue: #3a9fff
```

**What NOT to change:** Python logic, data fetching, session state, form handling, navigation structure, secrets, database logic.

---

## SESSION 3 — CMD-2 WATCHTOWER PANEL (3rd tab)

Left panel:
- Slot capacity bar (5/6)
- 6-slot Watchtower table (ticker, class, thesis, trigger, status badge, removal date)
- Blackout calendar: ASTS ~Apr 11, BWXT ~Apr 14, KTOS ~Apr 17, ONDS Mar 25
- Next scorecard countdown (Q2 Jun 2026)

Right panel:
- Active SOs (SO-1 through SO-6)
- Satellite Priority Scores (DC-13): NBIS 30/30, KTOS 25/30
- ADV gate reference: FN 967,560 | BWXT 1,417,695

---

## GSD SESSION PROTOCOL (step mode only)

`/gsd auto` is banned — Commander-in-loop constraint. Step mode only:
1. `/gsd` → discuss
2. `/gsd` → plan
3. `/gsd` → execute (one step at a time, confirm before each)
4. `/gsd` → verify

---

## WHAT NOT TO BUILD

- No buy/sell signals of any kind
- No auto-execution, auto-order, or broker API integration
- No `/gsd auto` mode
- No features that bypass Commander's decision in the deployment chain
- Alert systems (future P7): informational only
- No `node_modules` / webpack in `command_center/`
- Never generate or modify `.streamlit/secrets.toml`

---

*END OF HANDOFF — Updated Mar 16, 2026*
