---
trigger: auto
description: Vault rebalancing, switch triggers, E-Class rules — doctrine_pe v1.6.0
---

# VAULT PROTOCOL — doctrine_pe v1.6.0 (POINTER)

**Authoritative source:** doctrine_pe.md (version 1.6.0)

## Vault Structure
The Vault is the passive/locked investment layer — Thai provident fund wrappers,
RMF/SSF/SSFA/SSFE products, and E-Class (liquid) ETFs.

## Asset Classes
- **E-Class (Tier 2 — liquid):** ETFs in switchable wrappers. Count toward Level 1 at ×0.80.
  Current E-class: SCBKEQTGE (EWY), SCBS&P500(E), SCBSEMI(E), SCBGOLDHE
- **SSF/RMF/SSFA/SSFE (Tier 3 — locked):** No cash redemption. Do NOT count toward Level 1.
  Rebalance via switch to E-class only.

## Switch Triggers (T1-T4)
A switch may only happen when at least one trigger fires:

- **T1 — Thesis Break:** ETF thesis break at master level (SOXX/VOO/QQQ/EWY/GLD)
- **T2 — Underperformance:** 12-month delta vs VOO > 15 percentage points below VOO
- **T3 — Concentration:** Any single exposure > 50% of vault total
- **T4 — Earnings Blackout:** No switch ±7 days from key earnings in underlying holdings

Switch decision requires Commander + Consensus Log entry. No auto-switch.

## Current Vault Allocation (portfolio_state v1.3.14)
- SOXX Semiconductors: 35.7% (cap: 50%)
- VOO S&P 500: 30.6%
- Global Tech RMF: 10.2% (🔒 locked)
- QQQ Nasdaq-100: 8.2% (🔒 locked)
- US Equity SSF: 6.2% (🔒 locked)
- EWY South Korea: 3.4% (E-class 💧)
- SET Thailand: 2.6% (🔒 locked)
- Gold: 0.6% (E-class 💧)

## E-Class Haircut Rule
E-class value × 0.80 = amount counted toward Level 1 (Tier 2).
This accounts for Thai fund pricing uncertainty and intra-day liquidity cost.

## Rebalance Review Schedule
- **Annual:** December each year (full 5-step procedure)
- **Pre-Flight trigger:** At ฿800,000 liquid — vault rebalance for Level 2 structure
- **Ad hoc:** Any switch trigger fires, or Level milestone reached
- **Level 2 vault review:** At ฿1,000,000 Level 1 completion
