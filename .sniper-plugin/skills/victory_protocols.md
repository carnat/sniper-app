---
trigger: auto
description: Victory protocols — DC-49, DC-50a/b, DC-56, DC-58 from doctrine_core Section 6.6
---

# VICTORY PROTOCOLS — doctrine_core v1.12.1 Section 6.6 (POINTER)

**Authoritative source:** doctrine_core.md Section 6.6 "Victory Protocols"

## DC-49 — Victory Target Declaration
Every Arsenal position must have a declared Victory Target at time of thesis establishment.
Victory Target = the price at which the position is considered "won" and partial exit is
evaluated. No Victory Target = position cannot be partially exited without /council approval.

Example: ANET Victory Target $200 (declared Mar 10, 2026).

## DC-50a — Partial Victory Exit
When price reaches Victory Target:
1. Sell 50% of position (Victory Sell)
2. Record Consensus Log entry with realized P&L
3. Remaining 50% becomes "house money" — hold until Thesis Break or Level 2 reallocation
Commander must explicitly declare Victory at the session — no auto-execution.

## DC-50b — Full Victory Exit
If thesis break fires within 30 days of Victory Target hit:
- Full exit permitted without /council (DC-50b override)
- Consensus Log entry still required
- Proceeds go to Ammo Reserve for next DCA cycle

## DC-56 — HWM Victory Lock
When portfolio reaches a new High Water Mark milestone (each ฿25K increment above HWM):
1. Lock the new HWM in Consensus Log
2. Evaluate Drawdown Freeze trigger (DC-59): if current liquid < HWM − 20%, freeze all DCA
3. Update Level 1 progress bar in Command Center

## DC-58 — Level Milestone Victory
When Level 1 target (฿1,000,000) is reached:
1. Mandatory Victory Session (full War Council)
2. Vault rebalance review (level-appropriate allocation)
3. Level 2 structure declaration (target TBD at session)
4. Pre-Flight review trigger at ฿800,000 (80% of Level 1)
