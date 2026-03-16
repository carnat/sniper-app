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
3. WATCHTOWER (CMD-2 — Session 3)
4. BACKTESTER (CMD-3 — Session 4, future)
