# SNIPER DOCTRINE CORE TRIPWIRES — v1.0.0-BETA.1

> Per-ticker thesis-break definitions, interpretation flags, and breach procedure.
> Extends doctrine_core Section 6.4.

**DOC VERSION:** v1.0.0-BETA.1
**STATUS:** BETA — Pending full council ratification
**LAST AMENDED:** Mar 16, 2026

---

## CHANGELOG (LAST 3 ENTRIES)

- **[Mar 16, 2026] v1.0.0-BETA.1:** Initial extraction from doctrine_core Section 6.4. Full per-ticker tripwire set documented. BETA pending cross-model validation.
- **[Mar 16, 2026] v0.9.9:** AAOI permanently removed (Rule 1 blacklist). PLTR thesis break added (Rule 3 exception confirmed).
- **[Mar 15, 2026] v0.9.8:** SNDK blacklist added. ANET victory target amended (INVEST-05).

---

## SECTION 0 — TRIPWIRE SYSTEM OVERVIEW

A tripwire is a pre-defined business-event condition that, when fired, triggers a mandatory /council session.

**Critical rule:** Price drops of -25% to -40% are NOT thesis breaks. Thesis breaks are BUSINESS EVENTS only.

**Tripwire breach procedure:**
1. IO flags breach at session open.
2. Mandatory /council trim [TICKER] session initiated.
3. Council recommendation + Commander decision required before any exit.
4. Do not panic-sell without Council session unless VIX >= 25 AND tripwires are simultaneously breached (dual-gate systemic exit).

---

## SECTION 1 — STANDARD TRIPWIRE CATEGORIES

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

---

## SECTION 2 — INTERPRETATION FLAGS

**FLAG A — IGNORE CONSENSUS:**
- Tickers: ASTS, ONDS
- Maximum pain accepted. High-conviction asymmetric bet.
- Normal analyst consensus does not trigger a review. Only position's own tripwires matter.
- Flag A positions do not participate in Core DCA Matrix.

**FLAG B — FINANCIALS PRIMARY, CONSENSUS SECONDARY:**
- Tickers: MU, TSM
- Standard Core position where financial metrics are primary signal.
- Tripwire breach triggers mandatory /council trim session before any further capital deployment.

**FLAG C — GAAP/ADJ FILTER REQUIRED:**
- Tickers: VST
- GAAP EPS figures require adjustment filter.
- Unrealized hedge losses excluded from operational assessment.
- Use Adj EBITDA as primary metric.

**NO FLAG (Standard Core):**
- Tickers: VRT, APH, ANET, FN
- Standard Core position.
- Tripwire breach triggers mandatory /council trim session before any further capital deployment.

---

## SECTION 3 — PER-TICKER TRIPWIRE DEFINITIONS

### VRT (Vertiv Holdings)
**Thesis:** Data center thermal management — secular growth from AI infrastructure buildout.
**Tripwires:**
- TW-1: Hyperscaler capex declines for 2 consecutive quarters
- TW-1: Book-to-Bill falls < 1.0 for two consecutive quarters
- TW-1: Major hyperscaler officially awards primary cooling contract to competitor

### MU (Micron Technology)
**Thesis:** HBM memory dominance in AI training and inference supply chain.
**Tripwires (compound condition):**
- TW-1: HBM ASP declines for 2 consecutive quarters AND a forward guidance cut is issued in the same or subsequent print as the second consecutive ASP decline quarter
- TW-1: NVIDIA switches primary HBM supplier away from MU/SK Hynix
- TW-1: Total company gross margin falls < 20% for 2 consecutive quarters due to legacy DRAM/NAND pricing deterioration
**Flag: B**

### APH (Amphenol)
**Thesis:** Critical connectors for defense, datacom, and AI infrastructure — diversified hardware dependency play.
**Tripwires:**
- TW-1: Total revenue growth (Organic + M&A) falls < 5% YoY for 2 consecutive quarters
- TW-1: Defense/datacom segment combined drops < 45% of total revenue mix
**Active watch:** ⚠️ CHINA WATCH — China revenue between 15-20%. Trajectory must remain flat or declining.

### ANET (Arista Networks)
**Thesis:** AI networking infrastructure — dominant hyperscaler switching vendor with high switching costs.
**Tripwires:**
- TW-1: MSFT or Meta publicly announce primary switching vendor change away from Arista
- TW-1: Public shift > 20% of networking spend to internal "white box" custom solutions within a single fiscal year
- TW-1: AI/cloud vertical drops < 60% of new bookings for 2 consecutive quarters
- TW-1: Margin floor permanently breaches < 60% due to aggressive cloud titan pricing leverage
- TW-1: Microsoft/Meta formally announcing structurally lower AI networking CapEx

### TSM (Taiwan Semiconductor Manufacturing Co.)
**Thesis:** Irreplaceable foundry — monopoly position in advanced node manufacturing.
**Tripwires:**
- TW-2: Formal US military repositioning in Taiwan Strait
- TW-2: Sustained Chinese naval quarantine/blockade physically halts wafer shipments for > 30 days
- TW-2: US government mandates divestment
**Active watch:** ⚠️ ISRAEL WATCH (via TSEM relationship) + LNG cliff — monitor May 2026 window.
**Flag: B**

### VST (Vistra Corp)
**Thesis:** Nuclear energy + data center power — behind-the-meter nuclear PPAs with hyperscalers.
**Tripwires:**
- TW-1: Hyperscaler (Meta, MSFT, AMZN) formally cancels or materially reduces a Long-Term PPA
- TW-1: Regulatory/construction delays push contracted nuclear capacity expansion > 18 months behind announced timeline
- TW-1: New hyperscaler behind-the-meter nuclear PPAs permanently re-price below $100/MWh baseline combined with two consecutive quarters of hyperscaler energy CapEx guidance cuts
**Yellow Watch trigger (INVEST-04):** Two consecutive quarters of hyperscaler energy CapEx guidance cuts OR forward baseline power pricing (ERCOT/PJM) dropping > 15% over a 6-month period. Action: DCA into VST frozen (Factor 3 = 0), Victory Protocol soft stop tightened to 1.25x cost. Yellow Watch is NOT a thesis break — it is a DCA freeze pending resolution.
**Flag: C**

### FN (Fabrinet)
**Thesis:** High-precision optical manufacturing — sole-source critical supplier for AI datacom transceivers.
**Tripwires:**
- TW-1: Lumentum or Nvidia formally shifts primary optical manufacturing contract to a competitor
- TW-1: Datacom revenue growth drops < 10% YoY for two consecutive quarters
- TW-1: Industry achieves mass commercialization of Co-Packaged Optics (CPO) that actively cannibalizes FN's high-speed pluggable transceiver volume

### ASTS (AST SpaceMobile)
**Thesis:** Satellite-to-cell broadband — asymmetric growth bet on direct-to-device satellite internet.
**Tripwires:** N/A. Max pain accepted. Exit trigger = 15% CAP RULE only.
**Flag: A** — Ignore consensus. Only position's own CAP RULE governs.

### ONDS (Ondas Holdings / Sentrycs)
**Thesis:** Counter-drone defense platform — Sentrycs anchor contract with US Army/DHS.
**Tripwires:**
- TW-2: Sentrycs loses US Army or DHS anchor contract
- Position violates 15% CAP RULE
**Flag: A** — Ignore consensus. Only above tripwires govern.

### PLTR (Palantir Technologies)
**Thesis:** AI-enabled defense intelligence platform — Rule 3 exception confirmed INVEST-11.
**Tripwires:**
- TW-1: Government/defense/intel revenue drops below 50% of total revenue for 2 consecutive quarters
- TW-2: Government debarment or loss of security clearances affecting primary platform access
- TW-3: Management misconduct resulting in DOD contract suspension

### BWXT (BWX Technologies)
**Thesis:** Nuclear fuel and reactor components for defense and advanced nuclear power.
**Tripwires:**
- TW-1: US Navy micro-reactor program cancellation or multi-year delay
- TW-2: Export control restriction on nuclear technology components affecting primary contracts
- TW-1: Nuclear fuel segment revenue growth drops below 5% YoY for 2 consecutive quarters

### TSEM (Tower Semiconductor)
**Thesis:** Specialty analog/mixed-signal foundry — niche node semiconductor manufacturing.
**Active watch:** ⚠️ ISRAEL WATCH — mandatory /thesis refresh required before execution.
**Tripwires:**
- TW-2: Israel conflict escalation resulting in fab operational disruption > 30 days
- TW-1: Specialty node revenue declines > 15% YoY for 2 consecutive quarters

---

## SECTION 4 — PERMANENT BLACKLIST

The following tickers are permanently banned from Arsenal and Watchtower under Alpha Filter Rule 1:

| Ticker | Reason | Confirmed Date |
|--------|--------|----------------|
| AXTI | China rule breach (Rule 1) | Pre-codification |
| SNDK | China revenue 27.7% (Rule 1) | Mar 15, 2026 |
| AAOI | China PP&E 45.4% + mfg revenue 44.9% (Rule 1) | Mar 16, 2026 |

---

## SECTION 5 — BREACH PROCEDURE (DETAILED)

**Step 1 — Identification**
IO flags the breach at session open with: "⚠️ TRIPWIRE FIRED — [TICKER] — [TRIPWIRE TYPE] — [DESCRIPTION]"

**Step 2 — Escalation**
/council trim [TICKER] session mandatory. IO does not make exit recommendation without Council session.

**Step 3 — Council Options**
Council presents Commander with three options:
- A) MAINTAIN — Updated thesis confirms breach is cyclical, not structural. Log reason. Reset thesis clock.
- B) PARTIAL TRIM — Reduce position to X% of current size. Preserve thesis, reduce risk.
- C) FULL EXIT — Thesis permanently broken. Exit full position. Log realized P&L.

**Step 4 — Commander Decision**
Commander selects A, B, or C. Decision logged in Consensus Log with date, ticker, and selected option.

**Step 5 — Execution (Commander only)**
Commander activates broker action per selected option. IO logs execution confirmation when Commander reports.

**Dual-Gate Systemic Exit (exception to Step 2-4 sequencing):**
If VIX >= 25 AND multiple tripwires breached simultaneously: IO may surface immediate exit recommendation to Commander without waiting for full Council session. Commander may act. Full Council session is logged retroactively.

---

*END OF DOCTRINE CORE TRIPWIRES v1.0.0-BETA.1*
