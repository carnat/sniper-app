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

## War Room architecture (CMD-4)
- Canvas 2D rendering engine with `requestAnimationFrame` game loop
- `WarRoom` IIFE module: `init()`, `start()`, `stop()`, `updateFromPriceData(d)`, `syncPrices()`
- `Agent` class: ticker, zone (arsenal/watchtower/sentinel/commander), state machine, pixel art rendering
- 28 agents created from existing `ARSENAL_CORE`, `ARSENAL_SAT`, `WT_CORE_ROSTER`, `WT_SAT_ROSTER` arrays
- Sector-based color palettes: semi, defense, space, network, energy, vault, commander, sentinel
- Regime-based visual effects: CSS overlay changes with VIX regime (GREEN/YELLOW/ORANGE/RED)
- Game loop starts only when WAR ROOM tab is active (performance optimization)
- Integrates with `fetchPrices()` via `WarRoom.updateFromPriceData(d)` call
- VIX Sentinel agent changes color palette with regime state
- All agents freeze visually when VIX ≥ 25 (ORANGE/RED regime)
