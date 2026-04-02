---
trigger: auto
description: Deployment rules — gate sequence Q1-Q4, DC-15, DC-16 EQF, DC-31, DC-32
---

# DEPLOYMENT RULES — doctrine_core v1.12.1 Section 6 (POINTER)

**Authoritative source:** doctrine_core.md Section 6 "Deployment Gate Sequence"

## Gate Sequence (Q1 → Q4)

### Q1 — VIX Gate (DC-15)
**Primary deployment gate.** SPY must be above its 200-day SMA AND VIX < 25.
- VIX ≥ 25: ALL new DCA deployment FROZEN (Freeze Protocol)
- VIX < 22 (GREEN): Full deployment enabled
- VIX 22-24 (YELLOW): Deployment permitted with caution
- VIX 25-30 (ORANGE): Freeze active — no new DCA
- VIX > 30 (RED): Hard freeze — Arsenal + Watchtower expansion prohibited

**Exceptions:** DC-05 carve-out allows pre-placed broker orders (already queued) to
execute without VIX gate. Commander must declare carve-out explicitly.

### Q2 — Thesis Gate
Target ticker must have a current /thesis review ≤ 90 days old.
If thesis age > 30 days: ⚠️ NEGLECT flag (DC-06).
If thesis age > 60 days: mandatory /council review before any deployment.

### Q3 — ADV Gate (DC-16 EQF — Equal Fill Factor)
Minimum entry requirement: last session volume ≥ 1.5× the 20-day ADV.
ADV gate is a signal gate, not a hard block — confirms market conviction on the day.
- 1.5× ADV = "green close with conviction" trigger
- Below 1.5×: defer to next green close that meets gate

### Q4 — Blackout Gate
No new DCA within ±30 calendar days of earnings.
Blackout window: earnings_date − 30d to earnings_date + 7d.
Standing orders queued before blackout window opens are permitted to execute.

## DC-31 — Position Sizing
DCA per cycle is fixed at the session-declared amount. No scaling up in a single session.
Maximum single position size: 20% of Arsenal portfolio value (hard cap).

## DC-32 — Tranche Logic
Multi-tranche positions:
- Tranche 1: initial entry at Q1-Q4 gate clear
- Tranche 2: ≥5% retrace from T1 entry price (not from current price)
- Each tranche is separately gate-checked on its execution day
