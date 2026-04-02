---
trigger: auto
description: Sniper Portfolio State — current positions, ammo, Watchtower, standing orders, matrix scores
---

# PORTFOLIO STATE v1.0.17-BETA.1

**DOC VERSION:** v1.0.17-BETA.1
**LAST EDITED:** Mar 16, 2026
**STATUS:** BETA — Cross-model validation pending Claude review.

---

## CHANGELOG (LAST 3 ENTRIES)

- **[Mar 16, 2026] v1.0.17-BETA.1:** Marathon session. AAOI permanently banned (China PP&E 45.4% + mfg revenue 44.9%) — removed from Watchtower. BWXT reclassified Core (not Satellite). DC-12/13/13b ratified. Thesis runs: TSEM/FN/APH/VRT/NBIS all CLEAR Mar 16. Matrix final: FN=BWXT=70pts (tied), APH 65pts. Deployment plan locked. ADV reference added. Consensus Log +14 entries. News watermarks updated. Watchtower 5/6 slots.
- **[Mar 15, 2026] v1.0.16-BETA.1:** Spinoff session. Consensus Log +3 entries (Gate Validation, Vault Review, Watchtower Scorecard Q1). Watchtower capacity footer added. Backtester report archived on device.
- **[Mar 15, 2026] v1.0.15-BETA.1:** BWXT + KTOS added to Watchtower (slots 5+6, both 4/4 Strong Conviction). KTOS classified Satellite. AAOI /thesis Mar 15 + ⚠️ CHINA WATCH (manufacturing). LITE CONSIDERED/NOT ADOPTED. SO-8 added. FN reference price removed (DC-11). Consensus Log +8 entries. News watermarks BWXT/KTOS/AAOI → Mar 15.

---

## SECTION 1 — THE ARSENAL (TIER 1 INDIVIDUAL STOCKS)

### THE CORE (ANCHORS)

| TICKER | SHARES | COST | THESIS REVIEW DATE | FLAG | NOTES |
|--------|--------|------|--------------------|------|-------|
| VRT | 0.148355 | $235.92 | Mar 16, 2026 | — | Vertiv. Yellow Watch active → F3=0. Matrix 3rd (tied 50pts). DC-06 Day 0: Mar 13. |
| ASTS | 9.3 | $74.50 | NOT SET | A | SpaceMobile. SO-1 fill PENDING broker — submit market/wide limit < Apr 11. No VIX gate applies (DC-05 pre-placed carve-out). Blackout ~Apr 11 (earnings May 11). DC-06 Day 0: Mar 13. |
| VST | 12 | $158.32 | Mar 10, 2026 | C | Vistra. Yellow Watch active. |
| MU | 3.062527 | $294.01 | Mar 15, 2026 | B | Micron. Thesis INTACT ✅. Mar 18 earnings mandatory session. SCBKEQTGE vault decision Mar 19. |
| APH | 8 | $136.95 | Mar 16, 2026 | — | Amphenol. ⚠️ CHINA WATCH 16.3% (Q4 2025, trajectory falling toward 15%). Earnings Apr 29. Runner-up 65pts. |
| ANET | 3 | $122.12 | Mar 10, 2026 | — | Arista. Victory Protocol target $200. |
| TSM | 0.08434 | $296.42 | Mar 16, 2026 | B | TSMC. ⚠️ ISRAEL WATCH / LNG cliff May window. Matrix 4th (tied 50pts). |
| ONDS | 1 | $9.05 | NOT SET | A | Tracking position only. SO-2 hold. Mar 25 earnings + Mistral merger. No capital expansion. DC-06 Day 0: Mar 13. |
| FN | 0 | — | Mar 16, 2026 | — | Fabrinet. Matrix Winner (tied 70pts). Deploy ฿7,000 at gate clear. DC-11: F2=0, no cost basis. DC-06 Day 0: Mar 13. |

**Arsenal Count: 10 holdings** (portfolio_state v1.3.14 — VRT, ASTS, VST, MU, APH, ANET, TSM, ONDS, FN, COHR)

**DC-06 NOTE:** NEGLECT flag (⚠️ NEGLECT) activates **Apr 12, 2026** for all NOT SET positions (Day 0 = Mar 13). Forced /council trim activates **May 12, 2026** if still NOT SET.

**FLAG LEGEND:**
- Flag A = IGNORE CONSENSUS (Satellite/Special — max pain accepted)
- Flag B = FINANCIALS PRIMARY (MU, TSM — tripwire breach triggers /council trim)
- Flag C = GAAP/ADJ FILTER REQUIRED (VST — use Adj EBITDA as primary metric)
- No Flag (—) = Standard Core (VRT, APH, ANET, FN — tripwire breach triggers /council trim)

### THAI DEFENSE

| TICKER | SHARES | COST | RULE |
|--------|--------|------|------|
| TISCO | 173 | ฿97.26 | YIELD <5% OR NPL >3.5% |
| ADVANC | 26 | ฿284.04 | HOLD → DIV APR 30 |

**CONDO:** Val ฿1.7M | Debt ฿1.5M | CF +฿2,700/mo → THE ANCHOR

---

## SECTION 2 — WATCHTOWER (TIER 2 BUY LIST)

All Watchtower positions: Alpha Filters 1-4 passed. Satellite cap: max 15% of total liquid portfolio per position.

| TICKER | DATE ADDED | THESIS | BUY TRIGGER | REMOVAL DATE | STATUS |
|--------|------------|--------|-------------|--------------|--------|
| PLTR | Mar 12, 2026 | Palantir. Defense AI OS. Passes INVEST-11 Rule 3 Defense AI Exception. Gov rev >50%, physical-world influence, no HW substitute. Alpha Filters: all 4 pass. Vault overlap: zero. | VIX < 25 + green close >= 1.5x 20-day ADV | Jun 12, 2026 | Waiting for trigger. |
| TSEM | Mar 12, 2026 | Tower Semiconductor. SiPho optical foundry monopoly. 5x SiPho capacity by Dec 2026. PH18 platform — Lightwave Logic development agreement confirmed Mar 11. Alpha Filters: all 4 pass. Vault overlap: minimal. /thesis CLEAR Mar 16. | VIX < 25 + green close >= 1.5x 20-day ADV | Jun 10, 2026 | ⚠️ ISRAEL WATCH — Israel facility shipment disruptions confirmed. SiPho moat intact but /thesis refresh required before first execution. |
| NBIS | Mar 12, 2026 | Nebius Group. AI IaaS / GPU cloud. $2B NVIDIA strategic investment. GTC 2026 partnership expanded. Gross margin 27%→71% in 3 quarters. Scaling to 1GW+ power capacity. Alpha Filters: all 4 pass. Vault overlap: zero. /thesis CLEAR Mar 16. | VIX < 25 + green close >= 1.5x 20-day ADV | Jun 10, 2026 | TRIGGER HIT Mar 12. Option C active — see SO-6. ⚠️ EXPIRY MAR 26. |
| BWXT | Mar 15, 2026 | BWX Technologies. Sole-source US Navy nuclear component manufacturer + commercial nuclear renaissance (steam generators, fuel assemblies, TRISO fuel). Medical radioisotopes optionality. Iran war = direct thesis accelerant — nuclear urgency globally surging. CORE CANDIDATE — competes in Core Matrix (not Satellite). Deploy ฿11,000 from ฿18,000 tax refund. /thesis CLEAR Mar 15. | VIX < 25 + green close >= 1.5x 20-day ADV | Sep 15, 2026 | Waiting for trigger. Matrix Winner (tied 70pts). ⚠️ BLACKOUT ~Apr 14 (earnings May 4). Execution window ~4 weeks. ADV gate: 1,417,695. |
| KTOS | Mar 15, 2026 | Kratos Defense. Jet-powered attritable UAVs (Valkyrie), hypersonic vehicles, propulsion systems. Purest hardware play on US DoD "affordable mass" doctrine. Iran war = active conflict validates thesis directly. Airbus partnership opens NATO market. SATELLITE — max 15% total liquid portfolio. /thesis CLEAR Mar 15. | VIX < 25 + green close >= 1.5x 20-day ADV | Sep 15, 2026 | Waiting for trigger. DC-13 Satellite Priority Score: 25/30. ⚠️ BLACKOUT ~Apr 17 (earnings May 7). ⚠️ SATELLITE CAP: /audit required at execution. |

**WATCHTOWER: 12/12 CORE SLOTS + 3/3 SATELLITE SLOTS** (doctrine_ops v1.15.1 — expanded from legacy 6-slot to 15-position roster).

**WATCHTOWER CAPACITY (doctrine_ops v1.15.1):**
- **Core (12/12):** TSEM, BWXT, MOD, NBIS, FORM, ENTG, ONTO, LITE, QRVO, PLTR, KTOS, SKYT
- **Satellite (3/3):** RKLB, SATL, PL
- Next Scorecard review: **Jun 2026** (Q2 window opens Jun 1)
- Carry-forward candidates: AVAV, GEV, VSAT

---

## SECTION 3 — STANDING ORDERS

| SO# | DATE | DESCRIPTION | STATUS |
|-----|------|-------------|--------|
| SO-1 | Prior | ASTS: Fill +0.7 shares to reach 10.0 target. ~฿1,971.06. Market order or wide limit order (±5% of current price). Submit to broker BEFORE Apr 11 blackout opens. No VIX gate — pre-placed order carve-out (DC-05) applies. IO confirms submission timestamp predates Apr 11. | PENDING BROKER SUBMISSION CONFIRMATION |
| SO-2 | Prior | ONDS: Hold 1 tracking share. Monitor Mar 25 earnings + Mistral merger resolution. No capital expansion until smoke clears. | ACTIVE WATCH |
| SO-3 | Prior | Verify exact ASTS blackout date. | RESOLVED ✅ — ~Apr 11, 2026 (earnings May 11) |
| SO-4 | Prior | MU: Run /thesis before Mar 18 earnings. | RESOLVED ✅ — /thesis completed Mar 15, 2026. Thesis INTACT. |
| SO-5 | Prior | FN: Monitor for green close to reactivate DCA targeting. | ACTIVE WATCH — Matrix Winner. Gate: VIX<25 + green + ≥1.5x ADV (967,560). |
| SO-6 | Mar 12, 2026 | NBIS Option C (switched from Option A Mar 13, 4/4 Council). Tranche 1: Deploy ฿12,014.47 at next qualifying gate (VIX<25 + green close + ≥1.5x ADV). Tranche 2: ฿12,014.47 held for ≥5% retrace from entry. If no retrace within 30 days of Tranche 1 execution: redirect to next Matrix cycle. ⚠️ EXPIRY MAR 26: If VIX remains frozen past Mar 26, Commander decision required — extend or redirect Tranche 1. | AWAITING GATE — ⚠️ EXPIRY MAR 26 |
| SO-7 | Mar 15, 2026 | APH: Confirm China revenue % for DC-09 Watch Tier assessment. | RESOLVED ✅ — Mar 16, 2026. China revenue 16.3% Q4 2025. ⚠️ CHINA WATCH active. |
| SO-8 | Mar 15, 2026 | AAOI: Confirm China PP&E % from FY2025 10-K geographic notes before first execution. | RESOLVED ✅ — Mar 16, 2026. China PP&E 45.4% (FY2024 10-K Note R: $107.6M/$237.4M). China mfg revenue 44.9%. BOTH thresholds breached. PERMANENT BAN activated. AAOI removed from Watchtower. |

---

## SECTION 4 — DASHBOARD

**HIGH WATER MARK:** ฿287,218.16 — ARMED (established Mar 12, 2026)
**DRAWDOWN FREEZE:** Trigger at ฿229,774.53 (20% below HWM) — ARMED
**ACTIVE LEVEL:** LEVEL 1

### AMMO RESERVE

- **฿12,014.47** — TRANCHE 1. NBIS Option C first entry — deploy at next qualifying gate (VIX<25 + NBIS green + ≥1.5x ADV). ⚠️ Expiry Mar 26.
- **฿12,014.47** — TRANCHE 2. NBIS Option C pullback reserve — hold for ≥5% retrace from Tranche 1 entry price.
- **฿1,971.06** — ALLOCATED. ASTS SO-1 fill (pending broker submission).
- **฿18,000.00** — TAX REFUND INCOMING (not yet landed). Earmarked: ฿11,000 BWXT / ฿7,000 FN. Declare at session when received in account.

**DEPLOYMENT STATUS:** GATE FROZEN — VIX 27.19 (Mar 13). All Core/Watchtower deployment blocked. ASTS SO-1 independent of VIX gate.

**FX ZONE:** THB=X 32.42 — Zone B (32.00–36.00, normal deployment)

### CORE MATRIX SCORES (Mar 16, 2026 — FINAL)

| RANK | TICKER | F1 | F2 | F3 | F4 | TOTAL | NOTE |
|------|--------|----|----|----|----|-------|------|
| 🥇 | FN | 40 | 0 | 20 | 10 | 70 | Matrix Winner (tied). Deploy ฿7,000. |
| 🥇 | BWXT | 40 | 0 | 20 | 10 | 70 | Matrix Winner (tied). Deploy ฿11,000. BWXT wins tiebreak: blackout Apr 14 + higher ADV gate. |
| 🥈 | APH | 20 | 15 | 20 | 10 | 65 | Runner-up. |
| 🥉 | VRT | 30 | 10 | 0 | 10 | 50 | Tertiary (tied). Yellow Watch → F3=0. |
| 4th | TSM | 30 | 5 | 10 | 5 | 50 | Tied 3rd, loses on share count. |
| 5th | ANET | 20 | 10 | 10 | 5 | 45 | |
| 6th | MU | 10 | 0 | 10 | 5 | 25 | |
| 7th | VST | 0 | 10 | 0 | 3 | 13 | Yellow Watch → F3=0. |

**TIE-BREAK FN vs BWXT (DC-13 addendum):** BWXT wins — earlier blackout (Apr 14 vs none) + higher ADV gate threshold (1,417,695 vs 967,560). Option C resolution: both deploy from ฿18,000 split. ฿11,000 BWXT / ฿7,000 FN.

**GATE STATUS:** ❌ FREEZE — VIX 27.19. Sweep required Mar 18 session.

### ADV REFERENCE TABLE (calculated Mar 16, 2026)

| TICKER | 20-DAY ADV | 1.5× GATE THRESHOLD | NOTE |
|--------|------------|---------------------|------|
| FN | 645,040 | 967,560 | Lower bar — triggers on moderately active days |
| BWXT | 945,130 | 1,417,695 | Higher bar — requires catalyst or elevated sentiment days |

### SATELLITE PRIORITY SCORES (DC-13 — Watchtower Satellites only)

| RANK | TICKER | S1 | S2 | S3 | TOTAL | NOTE |
|------|--------|----|----|----|-------|------|
| 🥇 | NBIS | 10 | 10 | 10 | 30 | Triggered + expiry Mar 26 + NVIDIA re-rating + 0 shares |
| 🥈 | KTOS | 5 | 10 | 10 | 25 | Blackout Apr 17 + Iran war accelerant + 0 shares |

BWXT/PLTR/TSEM: Core candidates — Core Matrix governs. Arsenal Satellites (ASTS/ONDS): SO-governed per DC-13 scope.

**Aggregate Satellite sleeve:** ~0% of liquid portfolio (no Satellite positions held yet). 40% limit: CLEAR ✅

---

## SECTION 5 — RADAR

**MELI:** INVEST-12 CLOSED. No action taken.

---

## SECTION 6 — NEWS WATERMARK

| TICKER | LAST FETCH | CONDITIONAL RULE |
|--------|------------|-----------------|
| FN | Mar 16, 2026 | — Fetch only if new articles since watermark |
| ONDS | Mar 13, 2026 | — Fetch only if new articles since watermark |
| ASTS | Mar 14, 2026 | ⚠️ ACTIVE while SO-1 pending. Conditional watermark — IO News Gate applies. Remove when SO-1 resolved. |
| MU | Mar 14, 2026 | — Mandatory fetch Mar 18 earnings day |
| NBIS | Mar 16, 2026 | — Active watch — Option C Tranche 1 active, expiry Mar 26 |
| VRT | Mar 16, 2026 | — Fetch only if new articles since watermark |
| VST | Mar 14, 2026 | — Yellow Watch active — fetch on new catalyst only |
| ANET | Mar 14, 2026 | — Fetch only if new articles since watermark |
| TSM | Mar 16, 2026 | — ⚠️ ISRAEL WATCH / LNG cliff — fetch on new geopolitical signal |
| BWXT | Mar 16, 2026 | — Fetch only if new articles since watermark |
| KTOS | Mar 15, 2026 | — Fetch only if new articles since watermark |

---

## SECTION 7 — CONSENSUS LOG

| DATE | TYPE | DECISION | AUTHORITY | STATUS | REOPEN TRIGGER |
|------|------|----------|-----------|--------|----------------|
| Mar 12 | Watchtower Inclusion | PLTR approved for Tier 2 via Defense AI exception. Trigger: VIX<25 + green close >=1.5x ADV. Removal Jun 12, 2026. INVEST-11 resolved. | Commander/IO | 🔒 CLOSED | PLTR trigger hit OR removal date (Jun 12, 2026) |
| Mar 12 | Watchtower Inclusion | TSEM and NBIS approved for Tier 2. Standard 1.5x ADV triggers. Removal Jun 10, 2026. | Commander/IO | 🔒 CLOSED | Trigger hit OR removal date (Jun 10, 2026) |
| Mar 12 | NBIS Deployment Override | FN aborted this cycle. Ammo ฿24,028.94 redirected to NBIS. Missed Entry Option A selected initially. Buy alert $106.40. 10-day expiry. | Commander | 🔒 CLOSED | NBIS hits $106.40 OR alert expires (Mar 26, 2026) |
| Mar 12 | TSM Thesis Review | TSM thesis intact. Review date reset Mar 12, 2026. | IO | 🔒 CLOSED | 90 days elapsed OR new earnings |
| Mar 13 | D-01 FX Zone Pointer | FX Zone A/B/C table removed from doctrine_ops. Canonical home confirmed doctrine_core Section 6.2. Ratified 4/4. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | D-02 /thesis all Gate | IO Confirmation Gate added to /thesis all. Token cost + time estimate required before 9-ticker sweep. Ratified 4/4. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | D-03 ASTS Watermark | Conditional watermark for ASTS while SO-1 fill active. IO News Gate applies. Watermark removed on SO-1 resolution. Ratified 4/4. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | NBIS Missed Entry Switch | Option A ($106.40) switched to Option C. 4/4 STRONG CONVICTION. NVIDIA $2B = structural re-rating. Tranche 1: ฿12,014.47 on gate clear. Tranche 2: ฿12,014.47 for ≥5% retrace. | Commander/Council | 🔒 CLOSED | Tranche 1 executed OR ammo redirected next cycle |
| Mar 13 | DC-01 DCA Matrix Factors | Factor scoring brackets F1/F2/F3/F4 embedded in doctrine_core Section 6.5. 11/11 INVESTIGATE. Commander ratified. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | DC-02 Heartbeat Consequence | Missed heartbeat consequence clause added. VIX>=25 + 48h no slash command → all Arsenal ⚠️ HEARTBEAT MISSED, F1=0, deployment blocked. 8/11 INVESTIGATE. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | DC-03 PE Integrity Check | PE Section 5: verify COST column + Thai Defense block before generating portfolio_state at /close. 11/11 INVESTIGATE. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | FLAG Column Correction | VRT/APH/ANET/FN corrected to No Flag (—). VST corrected to Flag C. Tier A. | IO | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | DC-04 BETA Upload Alert | BETA alert added to Rule 5. IO flags pending BETA files at every session open. 8/11 Council. Tier B. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | DC-05 Blackout Protocol | Section 6.10 Blackout Protocol added. Pre-placed orders valid during blackout if IO confirms pre-blackout submission timestamp. Tier A — Commander waiver logged. 8/11 Council. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | DC-06 NEGLECT Escalation | NOT SET/STALE escalation rule added. 30-day NEGLECT flag, 60-day forced /council trim. Day 0 = Mar 13 for all existing NOT SET. Elder dissent on 60-day threshold on record. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | DC-07a F3 AND-Logic | F3 Strong Tailwind requires ALL 3 conditions simultaneously (earnings beat + guidance raise + sector acceleration). Partial = Moderate (10pts). Tier B. 9/11 Council. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | DC-07b Session Clock | Session clock for DC-02 defined. Qualifying commands: /brief /check /dca /thesis /review /audit /council /close. Casual queries excluded. Tier B. 8/11 Council. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | DC-07c Amendment Tier Routing | Amendment Validation Tier Routing: Tier S (COMET, 4-of-4), Tier A (cooling + waiver), Tier B (same-session OK). Macro dissent on cooling period on record. 9/11 Council. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 13 | DC-08 Model Council | Model Council Protocol. Roster: Claude (primary), Gemini (voting), Grok (voting), GPT-5.4/Comet (voting), Nemotron (IO validator). Thresholds: 4-of-4 Tier S, 3-of-4 Tier A. Handoff template pending /council doctrine. | Commander | 🔒 CLOSED | /council doctrine session to finalize DC-08 |
| Mar 15 | Rule 1 Blacklist — SNDK | SanDisk (SNDK) permanently blacklisted. China revenue 27.7% FY2025 10-K. | IO | 🔒 CLOSED | China revenue confirmed <20% for 2 consecutive annual filings |
| Mar 15 | DC-09a — China Watch Tier | China Watch Tier (15% trigger) added to Rule 1. Visibility flag only — no deployment block. Commander waiver on cooling granted. Strong Conviction 4/4. Bear dissent on complexity creep on record. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 15 | DC-09b — Grandfathering Clarification | Grandfathered positions >20% China revenue require formal Council verdict logged before next deployment cycle. No auto-exit. Tier B. 4/4. | Commander/Council | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 15 | DC-09a Commander Waiver | Cooling period waived for DC-09a. APH ~20-21% China; /thesis APH imminent; watch tier required before next DCA. | Commander | 🔒 CLOSED | — |
| Mar 15 | BWXT Watchtower Inclusion | BWXT approved Tier 2. 4/4 Strong Conviction. /thesis CLEAR. Three tripwires defined. Trigger: VIX<25 + green >=1.5x ADV. Removal Sep 15, 2026. | Commander/Council | 🔒 CLOSED | BWXT trigger hit OR Sep 15, 2026 |
| Mar 15 | KTOS Watchtower Inclusion | KTOS approved Tier 2. 4/4 Strong Conviction. SATELLITE classification. Three tripwires defined. Trigger: VIX<25 + green >=1.5x ADV. Removal Sep 15, 2026. | Commander/Council | 🔒 CLOSED | KTOS trigger hit OR Sep 15, 2026 |
| Mar 15 | LITE Rejected | LITE CONSIDERED / NOT ADOPTED. 1/4 Triads. FN/LITE inverse CPO correlation. Re-entry if FN exits OR LITE CPO >=40% of total. | Commander/Council | 🔒 CLOSED | FN exits Arsenal OR LITE CPO >=40% |
| Mar 15 | AAOI /thesis — Rule 1 Partial Pass | AAOI revenue test PASS (<10%). PP&E unconfirmed. SO-8 issued. ⚠️ CHINA WATCH (manufacturing) flag active. | IO/Commander | 🔒 CLOSED | SO-8 resolved |
| Mar 15 | DC-10 — /Thesis Early Suspension | IO must complete Alpha Filter pre-screen from get_stock_info before Calls 2-6. Blocking condition suspends run. Tier B. | Commander | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 15 | DC-11 — Zero-Share Factor 2 = 0 | Zero-share Core positions: Factor 2 = 0. No reference prices. FN reference $586.90 deleted. Tier B. | Commander | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 15 | GATE VALIDATION — Backtester v1.1 | 3-year backtest confirms VIX threshold (25) and green close filter calibrated correctly. No doctrine changes. Report archived on device. | Commander | 🔒 CLOSED | Re-run at Level 2 or major market structure change |
| Mar 15 | VAULT REVIEW — Blueprint Run | 5-step vault optimizer run. Zero triggers fired. SCBKEQTGE deferred to Mar 19 (post-MU earnings). QQQ remnants long-horizon watch. Vault Optimizer integrated into doctrine_core Section 6.9. | Commander | 🔒 CLOSED | Mar 19 (SCBKEQTGE) OR Dec 2026 annual OR 12-month underperformance trigger |
| Mar 16 | DC-12 — Two-State File Status Protocol | DRAFT state retired. BETA = uploaded + active + pending validation. LIVE = reviewed + ratified. DC-04 detection shifts to _BETA filename suffix. STATUS line informational only. BETA→LIVE gate: Claude integrity check → Commander ratifies → PE generates LIVE file. Tier B. | Commander | 🔒 CLOSED | /council doctrine only |
| Mar 16 | External Review — OpenAlice | OpenAlice (TraderAlice/OpenAlice) CONSIDERED / NOT ADOPTED. File-driven reasoning architecture validates Sniper design. Guard pipeline, Brain/commit model, EventLog format noted as context for DC-09 XML queue. No doctrine changes. | Commander | 🔒 CLOSED | DC-09 XML investigation session |
| Mar 16 | TSEM Israel Watch | TSEM Israel operations disruption — mature-node order rerouting to UMC/PSMC confirmed. SiPho PH18 moat intact (non-replicable). /thesis refresh required before first execution. | IO | 🔒 CLOSED | /thesis TSEM completed OR TSEM removed from Watchtower |
| Mar 16 | Rule 1 Blacklist — AAOI | AAOI permanently blacklisted. FY2024 10-K Note R: China PP&E $107.6M/$237.4M = 45.4%. China-manufactured revenue $111.8M/$249.4M = 44.9%. Both thresholds breached simultaneously. Removed from Watchtower. Slot 6 OPEN. | IO/Commander | 🔒 CLOSED | China PP&E <20% AND revenue <20% for 2 consecutive annual filings |
| Mar 16 | SO-7 Resolved — APH China % | APH China revenue 16.3% Q4 2025 ($1.05B/$6.44B). FY2026 est. 14.7%. Trajectory falling. ⚠️ CHINA WATCH activated per DC-09a. SO-7 closed. | IO | 🔒 CLOSED | China revenue <15% for 2 consecutive annual filings |
| Mar 16 | /thesis FN — CLEAR | FN thesis CLEAR. Revenue +35.9% YoY Q4 FY26. Gross margin 12.2% stable. FCF TTM +$101.6M. Net cash $955.9M. Effectively debt-free. Thailand manufacturing = zero Hormuz exposure. 8B/1H/0S. Median target $548. All tripwires clear. | IO | 🔒 CLOSED | 90 days OR earnings May 4 |
| Mar 16 | /thesis APH — CLEAR | APH thesis CLEAR. Revenue +49% YoY. Gross margin 38.2%. FCF TTM $4.38B. CommScope CCS closed Jan 2026. Net debt $4.4B declining at $4.4B FCF pace. 10B/4H/1SS. Median target $169. ⚠️ CHINA WATCH 16.3%. All tripwires clear. | IO | 🔒 CLOSED | 90 days OR earnings Apr 29 |
| Mar 16 | /thesis VRT — CLEAR | VRT thesis CLEAR. Gross margin 33.7%→38.9% over 4Q. FCF TTM $1.89B accelerating. Net debt $1.18B declining. S&P 500 addition Mar 2026. BYOP&C Alliance launched. 6SB/16B/3H/0S. Median target $280. ⚠️ YELLOW WATCH remains active → F3=0. All tripwires clear. | IO | 🔒 CLOSED | 90 days OR earnings Apr 29 (Yellow Watch review) |
| Mar 16 | /thesis TSEM — CLEAR | TSEM thesis CLEAR. Lightwave Logic PH18 deal confirmed. OFC 2026 presenter. Revenue +18.4% FY2026 consensus. Net cash $1.06B. FCF inflecting positive. 4B/2H/0S. ⚠️ ISRAEL WATCH active — /thesis refresh required before first execution. All core tripwires clear. | IO | 🔒 CLOSED | /thesis refresh before execution |
| Mar 16 | /thesis NBIS — CLEAR | NBIS thesis CLEAR. Gross margin 27%→71% in 3Q (GPU utilization flywheel confirmed). Revenue +355% YoY Q3 2025. Cash $4.79B, runway 4.6 years. NVIDIA GTC 2026 partnership expanded. 9B/2H/0S. Option C Tranche 1 authorized. 17.4% short interest (Flag A — accepted). All tripwires clear. | IO/Commander | 🔒 CLOSED | 90 days OR earnings Apr 29 |
| Mar 16 | BWXT Reclassified Core | BWXT reclassified from Satellite to Core Watchtower candidate. Competes in Core Matrix (F1=40, F2=0, F3=20, F4=10 = 70pts — tied Matrix Winner). Deploy ฿11,000 from ฿18,000 tax refund. ADV gate 1,417,695. DC-13 Satellite Priority Score does not apply. | IO/Commander | 🔒 CLOSED | Rule change via /council doctrine only |
| Mar 16 | /thesis BWXT — CLEAR (Core Matrix) | BWXT earnings beat confirmed (EQ growth +30.9% YoY). 9 upward revisions FY2026 EPS in 30 days. Sector acceleration: Iran war nuclear urgency + TD Cowen initiation $230 target + surging backlog. All 3 F3 AND conditions met → F3=20pts. Matrix Winner tied 70pts. All tripwires clear. | IO | 🔒 CLOSED | 90 days OR earnings May 4 |
| Mar 16 | DC-13 — Satellite Priority Scoring Tier | DC-13 ratified. Satellite Priority Score (30pts max): S1 Execution Urgency (10), S2 Thesis Conviction Delta (10), S3 Concentration Headroom (10). Scope: Watchtower Satellites only (KTOS, NBIS). Core Watchtower candidates (BWXT, PLTR, TSEM) and Arsenal Satellites (ASTS, ONDS) excluded. Aggregate Satellite sleeve limit 40% of total liquid portfolio added. Tier B. | Commander | 🔒 CLOSED | /council doctrine only |
| Mar 16 | DC-13b — Arsenal Satellite Expansion Gate | Post-fill-target capital expansion for Arsenal Satellites requires: (a) fresh /thesis CLEAR, (b) /audit confirms 15% cap headroom, (c) explicit Commander authorization in Consensus Log. Tier B. | Commander | 🔒 CLOSED | /council doctrine only |
| Mar 16 | DC-13 Core Tie Tiebreaker + Split Proportion | DC-13 addendum: Tied Core Winners with fresh capital split — earlier blackout gets ≥55%. If blackouts within 7 days or neither has blackout: higher ADV gate threshold gets larger allocation. If still tied: Commander discretionary + logged. Application: BWXT ฿11,000 (Apr 14 blackout + ADV gate 1,417,695) / FN ฿7,000 (no blackout + ADV gate 967,560) of ฿18,000 refund. Tier B. | Commander | 🔒 CLOSED | /council doctrine only |
| Mar 16 | Green Close Trigger Clarification | Gate Q2 green close = single-day confirmation. One green close + ≥1.5x ADV same session = deploy next open. Two-day confirmation NOT required. Volume ≥1.5x ADV is the institutional confirmation signal. Extension check (>10%) is the safety valve. Applies to all Core and Watchtower triggers. Tier B clarification. | IO/Commander | 🔒 CLOSED | /council doctrine only |
| Prior | INVEST-03 | ASTS 10-share target confirmed. Expansion to 20 rejected. | Commander/Council | 🔒 CLOSED | — |

---

## SECTION 8 — INVESTIGATIONS QUEUE

| ID | TICKER | STATUS | NOTES |
|----|--------|--------|-------|
| INVEST-01 through INVEST-10 | — | CLOSED | — |
| INVEST-11 | PLTR | CLOSED | Graduated Watchtower Mar 12. Defense AI Exception ratified. |
| INVEST-12 | MELI | CLOSED | No action taken. |

---

## SECTION 9 — PRICE CACHE

**STATUS: INVALIDATED** — sweep required Mar 18 session (MU earnings mandatory).

Last confirmed prices (Mar 13, 2026 close):
- VRT: $258.88 | ASTS: $86.34 | VST: $158.95
- MU: $426.13 | APH: $133.92 | ANET: $133.57
- TSM: $338.31 | ONDS: $10.16 | FN: $502.14
- BWXT: $194.13 | NBIS: $112.95
- VIX: 27.19 (FREEZE) | THB=X: 32.42 (Zone B)

---

## SECTION 10 — VAULT BENCHMARK PROTOCOL

POINTER: Vault benchmark protocol → doctrine_council (current version) Section 1.0.

Vault Settlement: 5/5 confirmed settled (Mar 12, 2026)

**SCBKEQTGE:** Decision deferred to Mar 19, 2026 (post-MU earnings Mar 18).

---

*END OF PORTFOLIO STATE v1.0.17-BETA.1*
