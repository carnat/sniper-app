# command_center/ — CLAUDE.md

Static HTML only. No Python. No server. No build step.
`index.html` is a single self-contained file — all CSS and JS inline.

## Data sources
- `../data/prices.json` — populated by GitHub Action daily at 09:00 BKK; read via `fetch('../data/prices.json')`
- `./private.json` — gitignored, local dev only; injected without PIN prompt
- `./private.enc.json` — AES-256-GCM encrypted, safe to commit; decrypted in-browser via PIN modal
- `../data/doctrine_state.json` — gitignored, live thesis/tripwire/blackout data; see `data/doctrine_state.example.json` for schema

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

## War Room architecture (CMD-4 + CMD-5 + War Council Parliament)
- Canvas 2D rendering engine with `requestAnimationFrame` game loop — **Metro Pixel Art** aesthetic
- Canvas expanded from 40×30 to **40×36** (576px base height) for parliament amphitheater
- `WarRoom` IIFE module: `init()`, `start()`, `stop()`, `updateFromPriceData(d)`, `syncPrices()`
- `Agent` class: ticker, zone (arsenal/watchtower/sentinel/commander/**council**), state machine, pixel art rendering
- 28 stock agents + **16 council agents** + Bear veto = **45 total agents**
- Sector-based color palettes: 25 entries (8-color: body/head/hi/dk/skin/glow/eye/accent)
- **Metro visual effects**: animated circuit-trace floor, parallax cityscape, neon zone borders, dithered shadows, CRT glitch bars, regime-driven atmosphere (green data-rain, amber sweep, orange glow, red flash), data-flow pipes between zones
- Regime-based visual effects: CSS overlay changes with VIX regime (GREEN/YELLOW/ORANGE/RED)
- Game loop starts only when WAR ROOM tab is active (performance optimization)
- Integrates with `fetchPrices()` via `WarRoom.updateFromPriceData(d)` call
- VIX Sentinel agent changes color palette with regime state
- All arsenal/watchtower agents freeze visually when VIX ≥ 25 (ORANGE/RED regime)

## War Council Parliament (16 members × 4 Triads + 2 floating + Bear)
Amphitheater layout with 4 Triad clusters across rows 24-33:

| Triad | Members | Conviction | Audit Domain |
|-------|---------|------------|-------------|
| **THESIS** (row 25, left) | ELDER, SA (Sector Analyst), CA (Capital Allocator) | 2/3 | Thesis age/neglect, F3 SMA checks, concentration |
| **TIMING** (row 25, right) | ROBOT, MACRO, CLOCK | 2/3 | VIX regime, ADV gates, SPY 200DMA, blackout windows |
| **SAFETY** (row 28, left, 4 seats) | RISK, ARCH, PSY, IENG | 3/4 | Tripwires, WT capacity, behavioral flags, removal dates |
| **EXT INTEL** (row 28, right) | IO, GEO, DSA (Demand-Side) | 2/3 | News watermark, China/Israel/LNG watches, demand signals |
| **FLOATING** (row 31) | REG (Regulator), TF (Tech Forecaster) | non-voting | Compliance scan, domain ticker triggers |
| **BEAR** (row 33, center) | BEAR | independent veto | Triggers when 3+ critical findings across all Triads |

- Audit functions: `thesisTriadAudit()`, `timingTriadAudit()`, `safetyTriadAudit()`, `extIntelTriadAudit()` — per-member finding assignment
- `runTriadAudits(d)` called inside `updateFromPriceData()` on every `prices.json` refresh
- Speech bubbles rotate through findings every 4-6s (staggered), priority-sorted: critical→warning→info→ok
- **Consensus**: when 2-of-4 Triads flag the same ticker, `⚡ CONSENSUS` label + golden table glow
- **Conviction badges**: each Triad zone shows ✓ CONVICTION / ✗ NO CONVICTION / — PENDING
- Floating agents: Regulator patrols (drifts between zones), TF activates for domain tickers
- Bear: independent veto layer, shows 🐻 BEAR VETO when 3+ critical alerts detected
- Click/hover on any council agent shows tooltip with Triad affiliation, conviction status, NON-VOTING badge
- Each member has unique body style (hair, build, accessories) — 15 new BODY_STYLES entries

## Doctrine implementation status
- ✅ VIX regime bands (GREEN/YELLOW/ORANGE/RED)
- ✅ PRI gauge (5-factor composite)
- ✅ Thesis age clock (NEGLECT/STALE tracking)
- ✅ Blackout window detection
- ✅ War Council Parliament (16 members × 4 Triads + floating + Bear)
- ✅ Per-member audit findings with conviction calculation
- ✅ Consensus detection (2-of-4 Triads)
- ✅ doctrine_state.json data pipeline (schema + loader + fallback)
- ⚠️ DCA Matrix scores are hardcoded — F1-F4 not computed live
- ⚠️ Drawdown Freeze displayed but not enforced (no DCA blocking)
- ⚠️ Victory Protocol targets shown but 2x/3x automation missing
- ❌ Alpha Filter validation not automated
- ❌ Bear Restructure logic not implemented
- ❌ Dead Hand Clause not implemented
- ❌ Concentration limit enforcement missing
