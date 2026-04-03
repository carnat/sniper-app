# command_center/ — CLAUDE.md

Static HTML only. No Python. No server. No build step.
`index.html` is a single self-contained file — all CSS and JS inline.

## Data sources
- `../data/prices.json` — populated by GitHub Action daily at 09:00 BKK; read via `fetch('../data/prices.json')`
- `./private.json` — gitignored, local dev only; injected without PIN prompt
- `./private.enc.json` — AES-256-GCM encrypted, safe to commit; decrypted in-browser via PIN modal

## Dependencies (CDN only)
- Fonts: JetBrains Mono + Rajdhani via `fonts.googleapis.com`
- Charts: Chart.js via `cdnjs.cloudflare.com` only

## Constraints
- Do not add `node_modules`, `package.json`, `webpack`, or any build tooling
- Do not add Python files here
- All state is hardcoded JS with `prices.json` overlay — no backend calls
- Never commit `private.json` (gitignored)
- `private.enc.json` is safe to commit — regenerate with `python scripts/encrypt_private.py`

## Tab structure
1. ARSENAL + STATUS (default) — tab ID: `arsenal`
2. VAULT HEALTH — tab ID: `vault`
3. WATCHTOWER — tab ID: `watchtower` (CMD-2 — ✅ Session 3)
4. NEWS — tab ID: `news` (CMD-2 — ✅ Session 3, filterable feed: ALL/ARSENAL/WATCHTOWER)
5. BACKTESTER — tab ID: `backtester` (CMD-3 — ✅ Session 4)
6. WAR ROOM — tab ID: `warroom` (CMD-4/5 — ✅ Agent-based pixel art + Triad Council)

Keyboard shortcuts: `1`–`6` switch tabs, `R` refresh, `?` help overlay, `Esc` close

## Command Center v2 enhancements (CMD-6→11)
Features added via PR #16, WorldMonitor-inspired:

- **PRI Gauge (B1)** — SVG circular gauge, 0–100 composite from 5 factors: VIX regime (0-30), drawdown proximity (0-25), concentration risk (0-15), blackout count (0-15), thesis neglect (0-15). Color zones: GREEN <30, YELLOW 30-60, ORANGE 60-80, RED 80+. `calculatePRI()` + `renderPRI(d)`.
- **Signal Convergence (B2)** — 7 cross-signal rules: ELEVATED RISK, FREEZE/HARD FREEZE, DEPLOYMENT WINDOW, SPY BELOW 200DMA, THESIS NOT SET, EARNINGS BLACKOUT, VOLUME SURGE. `detectSignals()`.
- **Inline Sparklines (A2)** — `createSparkline(entry)` generates 28×12px SVG polylines from `change5d` data in Arsenal + Watchtower tables.
- **Ticker Tape (A4)** — Auto-scrolling horizontal bar below header, CSS `tape-scroll 45s` animation, pauses on hover.
- **News Sentiment Badges (B4)** — Keyword-based BULLISH/BEARISH classification + category tagging (EARNINGS/SECTOR/MACRO/DOCTRINE). `newsBadges()`.
- **Auto-Refresh Toggle (B8)** — 5-min polling with localStorage persistence + stale data badges (24h amber, 48h red).
- **Mobile Bottom Tab Bar (A6)** — Fixed bottom nav with emoji icons, `@media (max-width: 600px)`, `safe-area-inset-bottom`.
- **PanelManager (A1)** — Collapse/expand panels with localStorage state persistence.
- **CSS Semantic Tokens** — `--panel-radius`, `--font-xs`→`--font-xl`, `--shadow-card`, `--transition-fast/normal`.
- **Shared Helper** — `getBlackoutTickers()` used by both PRI and Triad Council audit engine.

## War Room architecture (CMD-4 + CMD-5)
- Canvas 2D rendering engine with `requestAnimationFrame` game loop
- `WarRoom` IIFE module: `init()`, `start()`, `stop()`, `updateFromPriceData(d)`, `syncPrices()`
- `Agent` class: ticker, zone (arsenal/watchtower/sentinel/commander/**council**), state machine, pixel art rendering
- 28 stock agents created from existing `ARSENAL_CORE`, `ARSENAL_SAT`, `WT_CORE_ROSTER`, `WT_SAT_ROSTER` arrays
- Sector-based color palettes: semi, defense, space, network, energy, vault, commander, sentinel
- **3 Triad Council agents**: DOCTRINE (purple), GATE (steel-blue), WATCHER (emerald) — zone `council`
- Regime-based visual effects: CSS overlay changes with VIX regime (GREEN/YELLOW/ORANGE/RED)
- Game loop starts only when WAR ROOM tab is active (performance optimization)
- Integrates with `fetchPrices()` via `WarRoom.updateFromPriceData(d)` call
- VIX Sentinel agent changes color palette with regime state
- All arsenal/watchtower agents freeze visually when VIX ≥ 25 (ORANGE/RED regime)

## Triad Council (CMD-5)
Three static agents at the bottom of the room (zone `council`) that continuously audit doctrine:

| Agent | Palette | Audit domain |
|-------|---------|--------------|
| DOCTRINE | doctrine (purple) | Alpha Filters, thesis age/neglect (DC-06), China Watch (DC-46), Israel Watch (DC-47), blackout windows |
| GATE | gate (steel-blue) | VIX regime (DC-15), ADV gate (DC-16), SPY vs 200DMA, heartbeat protocol |
| WATCHER | watcher (emerald) | DC-40 capacity, removal dates, tripwires (TW-1→TW-5), F3 SMA live validation |

- Audit functions: `doctrineAudit()`, `gateAudit()`, `watchtowerAudit()` — pure computation, no API calls
- `runTriadAudits(d)` called inside `updateFromPriceData()` on every `prices.json` refresh
- Speech bubbles rotate through findings every 6s, priority-sorted: critical→warning→info→ok
- **Consensus**: when 2-of-3 members flag the same ticker, `⚡ CONSENSUS` label appears at table center
- Click/hover on any Triad agent shows tooltip with full findings list + severity counts
- Room expanded from 26→30 rows to accommodate council zone
