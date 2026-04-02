---
trigger: auto
description: Thesis break / tripwire definitions — doctrine_core_tripwires v1.0.0
---

# TRIPWIRES — doctrine_core_tripwires v1.0.0 (POINTER)

**Authoritative source:** doctrine_core_tripwires.md (version 1.0.0)

## What is a Tripwire?
A tripwire is a pre-defined thesis-break condition for each Arsenal or Watchtower position.
When a tripwire fires, a mandatory /council session is triggered within 48 hours.
The Commander must decide: hold, reduce, or exit. No action = thesis abandoned by default.

## Standard Tripwire Categories

### TW-1 — Revenue / Thesis Break
Primary business thesis fails quantitatively:
- Revenue growth rate drops below sector median for 2 consecutive quarters
- Guidance cut > 10% of consensus
- Key contract / customer lost representing > 15% of revenue

### TW-2 — Regulatory / Geopolitical Shock
- New sanctions or export controls targeting primary revenue geography
- Government block on key customer or product line
- Conflict escalation affecting primary fab location

### TW-3 — Management / Governance Failure
- CEO or CFO departure in adverse circumstances (not planned succession)
- Restatement of financials
- Material fraud or regulatory investigation opened

### TW-4 — Capital Structure Trigger
- Dilutive equity offering > 10% of float without accretive use of proceeds
- Covenant breach or credit downgrade to below investment grade

### TW-5 — Concentration / China Watch Breach
- China revenue exposure crosses 20% threshold → disqualification
- Geopolitical watch flag escalates to active conflict impact on operations

## Per-Ticker Tripwire Notes (portfolio_state v1.3.14)
- **APH TW-5:** China at ~15.6% — trajectory must continue declining below 15% by next review
- **ONDS TW-1:** Mistral merger resolution + Q1 2026 revenue disclosure required
- **TSEM TW-2:** Israel conflict escalation — /thesis refresh required before execution
- **TSM TW-2:** LNG cliff + export control risk — monitor May 2026 window

## Response Protocol
1. Tripwire fires → immediate Consensus Log entry
2. /council session within 48 hours (24 hours if VIX ≥ 25)
3. Options: maintain (with updated thesis), partial trim, full exit
4. Decision recorded in Consensus Log — no silent holds
