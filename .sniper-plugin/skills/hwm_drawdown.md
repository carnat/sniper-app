---
trigger: auto
description: HWM tracking, Drawdown Freeze, DC-59 Amber Zone — doctrine_core v1.12.1
---

# HWM + DRAWDOWN — doctrine_core v1.12.1 Section 6.5 (POINTER)

**Authoritative source:** doctrine_core.md Section 6.5 "HWM and Drawdown Freeze Protocol"

## High Water Mark (HWM) Definition
The HWM is the highest recorded liquid portfolio value (Arsenal + Vault E-class × 0.80).
It is updated upward whenever liquid portfolio sets a new all-time high.
HWM is NEVER adjusted downward.

**Current HWM (portfolio_state v1.3.14):** ฿324,049

## Drawdown Freeze (DC-59)
If current liquid portfolio < HWM − 20%:
- All new DCA deployment is FROZEN immediately
- Only pre-placed broker orders (DC-05 carve-out) may execute
- Freeze lifts when liquid portfolio recovers to HWM − 10% (Amber Zone exit)

**Freeze trigger threshold:** HWM × 0.80

## DC-59 Amber Zone
The Amber Zone is the range between HWM − 20% and HWM − 10%:
- Deployment is RESTRICTED but not fully frozen
- Maximum DCA: 50% of normal cycle amount
- Thesis reviews are mandatory before any deployment in Amber Zone
- Amber Zone lifts when liquid > HWM − 10%

## HWM Milestone Lock (DC-56)
At each ฿25K increment above the previous HWM:
1. Update HWM in Consensus Log
2. Update Level 1 progress bar
3. Evaluate DC-59 drawdown threshold
4. Commander declares new HWM at session

## Current Status (portfolio_state v1.3.14)
- HWM: ฿324,049 (set — date to be confirmed at next session)
- Regime: YELLOW/ORANGE (VIX thawed, SPY below 200DMA)
- DC-59 freeze threshold: ฿324,049 × 0.80 = ฿259,239
- Amber Zone: ฿259,239 → ฿291,644

## Heartbeat Protocol
When VIX ≥ 25 (ORANGE/RED regime):
- Daily session REQUIRED (post-market review)
- 48-hour miss = HEARTBEAT MISSED → all Factor 1 scores set to 0 for next DCA cycle
- Cadence relaxes to weekly when VIX < 22 (GREEN regime)
