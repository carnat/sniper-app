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
1. ARSENAL + STATUS (default)
2. VAULT HEALTH
3. WATCHTOWER (CMD-2 — ✅ Session 3 complete)
4. BACKTESTER (CMD-3 — ✅ Session 4 complete)
5. WAR ROOM (CMD-4 — ✅ Agent-based pixel art visualization)

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
