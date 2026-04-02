---
trigger: auto
description: Watchtower management, DC-40 tiers, admission/removal — doctrine_ops v1.15.1
---

# WATCHTOWER — doctrine_ops v1.15.1 Section 0.4 (POINTER)

**Authoritative source:** doctrine_ops.md Section 0.4 "Watchtower Roster Management"

## Watchtower Purpose
The Watchtower is the pre-deployment staging area for Arsenal candidates.
Tickers on the Watchtower are being actively monitored for trigger conditions.
They have passed Alpha Filters (Rules 1-4) but are not yet in the Arsenal.

## Capacity (DC-40)
- **Core tier:** Up to 12 concurrent positions (S1/S2/S3 sub-tiers)
- **Satellite tier:** Up to 3 concurrent positions (SAT-1/SAT-2/SAT-3 sub-tiers)
- Total capacity: 15 positions

### Current Roster (doctrine_ops v1.15.1 / portfolio_state v1.3.14)

**Watchtower Core (12):**
TSEM, BWXT, MOD, NBIS, FORM, ENTG, ONTO, LITE, QRVO, PLTR, KTOS, SKYT

**Watchtower Satellite (3):**
RKLB, SATL, PL

## Tier Definitions
- **S1 (Core Tier 1):** Highest conviction — trigger imminent or recurring DCA active
- **S2 (Core Tier 2):** High conviction — waiting for gate clear
- **S3 (Core Tier 3):** Monitoring — thesis confirmed, queued for next scorecard
- **SAT-1/2/3:** Satellite tiers — deploy from Satellite capital only (DC-13 score ≥25 req'd)

## Trigger Conditions
Standard trigger: 1 green close with volume ≥ 1.5× 20-day ADV + VIX gate clear (< 25)
Exceptions:
- Pre-placed orders (DC-05 carve-out) bypass VIX gate
- Satellite triggers require DC-13 satellite score ≥ 25 in addition to standard gate

## Admission Criteria
1. Pass all Alpha Filters (Rules 1-4)
2. Scorecard session endorsement (≥ 2 Triad members)
3. Commander approval
4. Removal date set at admission (max 6 months unless extended by /council)

## Removal Triggers (DC-40)
Auto-removal on any one of:
1. Removal date reached with no trigger → free slot, no execution
2. Thesis break confirmed → immediate removal + Consensus Log
3. Core graduation → promoted to Arsenal, slot freed
4. Satellite deploy complete → hold as Arsenal position, Watchtower slot freed
5. Alpha Filter breach (China >20%, permanent ban) → immediate removal

## Blackout Windows
No new DCA during earnings blackout (±30d from earnings date).
Standing orders queued before window opens are exempt (DC-05).
