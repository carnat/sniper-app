# SNIPER DOCTRINE CORE — v1.12.1

> Do not reinterpret, paraphrase, or condense any rule below.

**DOC VERSION:** v1.12.1
**STATUS:** Current
**LAST AMENDED:** Mar 16, 2026

---

## CHANGELOG (LAST 3 ENTRIES)

- **[Mar 16, 2026] v1.12.1:** DC-13 — Satellite Priority Scoring Tier added to Section 6.7. DC-13b — Arsenal Satellite Expansion Gate added to Section 6.8 SATELLITE block. DC-13 Core Tie Tiebreaker + split proportion rule added to Section 6.5 Rule 2. Green close single-day confirmation stated explicitly in Section 6.2 Q2. BWXT reclassified Core. Tier B. Commander ratified Mar 16.
- **[Mar 15, 2026] v1.12.0:** Section 6.9 Vault Rebalancing Protocol expanded with full 5-step /vault_review procedure. Section 6.6 Watchtower Graduation Rule — POINTER added to Watchtower Scorecard quarterly protocol. Commander ratified Mar 15.
- **[Mar 15, 2026] v1.11.9:** DC-11 — Zero-Share Factor 2 = 0 Rule added to Section 6.5. Reference prices for zero-share positions prohibited. Commander ratified Mar 15.

---

## SECTION 0 — META

POINTER: Full operational behaviors → doctrine_ops (current version) Section 0.
POINTER: War Council protocol → doctrine_council (current version).
POINTER: Prompt Engineer protocol → doctrine_pe (current version).
POINTER: State data → portfolio_state (current version) [gitignored, local only].
POINTER: Vault data → vault_registry (current version) [gitignored, local only].

---

## SECTION 1 — ACTIVE LEVEL

**LEVEL 1: 1,000,000 THB Liquid.**

All buys serve LEVEL 1 first. No leverage. No margin. No derivatives.

---

## SECTION 2 — ASSET CLASSIFICATION

### TIER 1 — CORE (ANCHORS)
Individual US equities. High-conviction, long-term thesis. Scored via DCA Matrix (Section 6.5). Matrix Winner claims ammo each cycle.

Goal: Level 2 (Growth). Stop Loss: THESIS BREAK (Section 6.4) and VICTORY PROTOCOLS (Section 6.5). Trim only if a single asset > 40% of total liquid portfolio.

### TIER 1 — SATELLITE
Individual US equities outside Core Matrix. High-conviction special cases. Max allocation: 15% of total liquid portfolio per position. Stop Loss: None (-100% max pain accepted). Governed by: 15% Cap Rule + Section 6.2 Gate + War Council conviction. NOT scored by DCA Matrix. Require explicit Commander override to fund from Core ammo pool, OR fresh Tier 1 capital injection.

### TIER 1 — SHIELD
Thai defensive positions. Goal: Level 1 (Liquidity). Stop: Yield < 5% OR NPL > 3.5%.

### TIER 2 — WATCHTOWER
Qualified candidates awaiting entry trigger. Graduation requires: /thesis pass + Alpha Filter check (Section 5). Removal: trigger hit, removal date reached, or thesis tripwire breached.

Watchtower tickers carrying SATELLITE classification: governed by DC-13 Satellite Priority Scoring (Section 6.7).
Watchtower tickers WITHOUT Satellite classification (Core candidates): compete in Core DCA Matrix.

### TIER 2 — VAULT (PASSIVE)
Thai mutual funds and ETFs. Managed separately via /vault. Not traded tactically. Rebalanced per Section 6.9.

**E-CLASS EXCEPTION:** E-Class wrapper funds are LIQUID and SWITCHABLE. They count as TIER 2 (80% haircut) toward Level 1 target.

POINTER: Vault holdings and overlap → vault_registry (current version) [local only].

### CONCENTRATION DENOMINATOR RULE
All concentration measurements use **TOTAL LIQUID PORTFOLIO** as the denominator. Total liquid = (Tier 1 Cash) + (Tier 2 Assets × 0.80), per Section 7 formula. This explicitly includes E-Class vault funds at 80% haircut. Do NOT use US Arsenal alone as the denominator.

---

## SECTION 3 — NVDA EXCLUSION

NVDA is designated as a Vault Anchor. It is permanently restricted from direct Arsenal deployment due to extreme Vault index overlap (SCBSEMI + SCBS&P500 provide full exposure). We trade its hardware slipstream exclusively: TSM, FN, VST.

**This rule is CLOSED FOR DEBATE.** Reopen trigger: Vault SCBSEMI and SCBS&P500 both fully redeemed/switched away from NVDA exposure.

This rule has no override. It is unconditional.

---

## SECTION 4 — HIGH WATER MARK AND DRAWDOWN FREEZE

### HWM TRACKING
IO tracks portfolio peak value (High Water Mark) after each /audit. HWM resets upward only — never downward.

POINTER: Current HWM value → portfolio_state (current version) Section 4 [local only].

### DRAWDOWN FREEZE (THE SYSTEMIC DRAWDOWN FREEZE — ratified Mar 10, 2026 — INVEST-02)
The IO will track the HIGH WATER MARK — the highest recorded value of the Total Liquid Portfolio. If the Total Liquid value falls >= 20% below the High Water Mark, a SYSTEMIC FREEZE is activated. All DCA deployments and new asset purchases are BLOCKED, overriding the VIX gate. Selling operations remain active.

Freeze lift: /council full with explicit Commander authorization only. Commander cannot self-lift.

Freeze does not apply to /vault rebalancing.

**Freeze threshold formula:** HWM × 0.80
**Amber Zone formula:** HWM × 0.80 → HWM × 0.90 (restricted deployment, 50% of normal cycle)

---

## SECTION 5 — ALPHA FILTERS (ENTRY GATE — RULES 1 THROUGH 4)

All Arsenal and Watchtower candidates must pass all 4 filters. Filters are checked at /thesis time, not at execution.

A position already in the Arsenal is grandfathered unless a tripwire fires.

### RULE 1 — THE 20/20 CHINA RULE
Do not buy stocks that derive > 20% of total revenue from mainland China, OR hold > 20% of their physical manufacturing assets/facilities in mainland China. If either threshold is breached, the asset is permanently banned.

**Current permanent blacklist (Rule 1):**
- AXTI — permanently blacklisted (China rule breach)
- SNDK — permanently blacklisted (China revenue 27.7%, confirmed Mar 15, 2026)
- AAOI — permanently blacklisted (China PP&E 45.4% + China-manufactured revenue 44.9%, confirmed Mar 16, 2026)

**CHINA WATCH TIER (DC-09a — ratified Mar 15, 2026):** Any Arsenal or Watchtower position with confirmed China revenue between 15% and 20% is automatically flagged with ⚠️ CHINA WATCH status by IO at the next /thesis or /brief session. IO reports the current China revenue % and trajectory (rising/flat/falling). CHINA WATCH does not block DCA or deployment — it is a visibility flag only. If China revenue subsequently crosses 20%, the permanent ban activates immediately. CHINA WATCH clears automatically if China revenue falls and stays below 15% for two consecutive annual filings.

**GRANDFATHERING CLARIFICATION (DC-09b — ratified Mar 15, 2026):** The grandfathering clause protects existing Arsenal positions from being exited solely due to Alpha Filter breach at time of initial rule codification. However, if a grandfathered position subsequently has its China revenue confirmed above 20% at a /thesis run, IO must surface this to Council for explicit Commander review. The position is not auto-exited — but Council must deliver a formal verdict logged in the Consensus Log before the next deployment cycle for that ticker.

### RULE 2 — NO EUROPE (EXCHANGE LISTING)
US-listed securities only: NYSE or NASDAQ primary listing. No European primary listings, even if cross-listed in the US. ADRs are permitted if the underlying business passes Rules 1, 3, and 4.

### RULE 3 — HARDWARE / PHYSICAL-WORLD REQUIREMENT
No pure-play SaaS or software platforms without physical-world dependency. The business must have a material connection to physical infrastructure, hardware manufacturing, or real-world operational outcomes.

**EXCEPTION — 3-POINT DEFENSE AI QUALIFICATION TEST**
Ratified: INVEST-11, Mar 12, 2026. Authority: Commander + IO. Dual-AI validated (Gemini + Comet).

A software or AI platform MAY pass Rule 3 if ALL THREE of the following conditions are confirmed simultaneously:

- **Condition 1 — Revenue Composition:** Sovereign, defense, or intelligence revenue constitutes 50% or more of total company revenue on a trailing twelve-month basis.
- **Condition 2 — Physical-World Outcome Influence:** The platform directly influences kinetic or physical-world outcomes. Purely analytical, reporting, or forecasting functions do not qualify. The platform must be embedded in operational decision chains that result in physical actions.
- **Condition 3 — No Viable Hardware Substitute:** No viable hardware substitute exists for the platform's core function. If a hardware-only alternative could replicate the core capability, the exception does not apply.

If all 3 conditions are confirmed → asset passes Rule 3 regardless of software or SaaS classification.

**Confirmed passing example:**
PLTR (Palantir Technologies) — confirmed Mar 12, 2026.
- Condition 1: ~68% government/defense/intel revenue (> 50%). PASS.
- Condition 2: Platform directly embedded in kinetic military and intelligence operations. PASS.
- Condition 3: No hardware substitute replicates AIP/Gotham/Foundry integration. PASS.

### RULE 4 — VOLUME CONFIRMATION (EXECUTION GATE)
No execution without volume confirmation >= 1.5x 20-day average daily volume (ADV) on the entry candle.

This gate applies at execution time, not at Watchtower graduation.

A position may be Watchtower-qualified but remain BLOCKED if volume does not confirm on the intended entry day.

There is no override for this rule. If volume fails, wait for the next qualifying candle.

---

## SECTION 6 — THE DCA MATRIX AND DEPLOYMENT SYSTEM

### SECTION 6.0 — MONTHLY DEPLOYMENT PROTOCOL

**TIMING:** Executes on the last weekday of every month.

POINTER: Current ammo stack → portfolio_state (current version) Section 4 [local only].

**TARGET PRIORITY:**
1. **PRIMARY** — Winner of Section 6.5 DCA Targeting Matrix. Execute at planned size (or per Section 6.2 scaling rules).
2. **REMAINDER** — ASTS fill to target shares (verify fill target in portfolio_state before executing). **ZERO-REMAINDER RULE:** If ammo remaining after primary DCA allocation is insufficient to purchase at least 0.1 shares of ASTS, skip the ASTS fill order this cycle. Carry ASTS fill target to next deployment cycle. No partial ASTS purchase below 0.1 shares. Once ASTS reaches fill target, remainder held as USD Cash per USD Cash Holding Rule below.

**USD CASH HOLDING RULE:** Park in USD Money Market or USD wallet. Do not convert back to THB.

**ENTRY MECHANICS:** Governed strictly by Section 6.2 Universal Entry Protocol.

### SECTION 6.1 — PRIORITY HIERARCHY

The Core Matrix Winner holds absolute priority over Tier 2 Satellite expansion in every deployment cycle.

Capital parked for the Matrix Winner cannot be diverted to Satellites without a formal Missed Entry Protocol override (Section 6.3).

The override requires explicit Commander authorization and is logged in the Consensus Log.

### SECTION 6.2 — UNIVERSAL ENTRY PROTOCOL (DECISION TREE)

Run in order. A NO at any step stops deployment.

**STEP 1 — SYSTEMIC GATE**

Q1: Is VIX < 25?
- YES → proceed to Q2.
- NO → HOLD. Drawdown Freeze protocol. Stack ammo.

Q2: Did the target close green (positive) today on volume >= 1.5x its 20-day ADV?
- YES → proceed to Q3.
- NO → HOLD. Wait for confirmation candle. Do not chase red closes or low-volume days.

**GREEN CLOSE CONFIRMATION — SINGLE DAY (clarified Mar 16, 2026):** Gate Q2 requires one qualifying candle only: green close + volume ≥1.5x 20-day ADV on the same session. Execute at next session open. Two-day confirmation is NOT required and NOT doctrine. Volume ≥1.5x ADV IS the institutional confirmation signal. The >10% extension check (Q3) is the built-in safety valve against chasing.

Q3: Has the target risen more than 10% since the reference price?
- YES → EXTENDED. Invoke Missed Entry Protocol (Section 6.3).
- NO → GREEN-LIT. Proceed to Step 2.

**STEP 2 — FX GATE**

| THB=X Range | Zone | Action |
|-------------|------|--------|
| < 32.00 | Zone A | Deploy full ammo immediately |
| 32.00 to 36.00 | Zone B | Normal deployment — execute at planned size |
| > 36.00 | Zone C | Deploy only if Gate passes fully. Split remainder: 50% deploy → next Buy List target. 50% hold as THB Cash. |

**ABSOLUTE RULES**
1. Never place blind limit orders as entry vehicles.
2. A green day on a gapping target is not a confirmation unless volume >= 1.5x 20-day ADV.
3. Redirected ammo stays within the Buy List (portfolio_state Section 2).
4. The 10% extension threshold is measured against the price recorded at the most recent doctrine snapshot date or deployment decision, whichever is more recent.

### SECTION 6.3 — MISSED ENTRY PROTOCOL

Triggered when a target is EXTENDED (> 10% above reference price).

Commander selects one of the following options:

**OPTION C (PRIMARY) — FOMO HEDGE (Partial Entry):** Scale down to 50% of planned position size. Hold remaining 50% for a pullback entry. Do not add the second 50% until price retraces at least 5%.

**OPTION A (SECONDARY) — DUST SETTLE (Wait):** Set a MANUAL PRICE ALERT at 5% below the spike price. Buy nothing today. Ammo remains parked. Alert expires after 10 trading days. Re-run Section 6.2 gate when triggered. If 10 days pass without a fill: alert expires. Re-evaluate via Option B.

**OPTION B (FALLBACK) — ABORT:** Abort the strike entirely. Redirect ammo to second-highest Matrix scorer or hold USD Cash for next resupply. Leave target in Watchtower for next qualifying cycle.

Default if Commander issues no order: Option C.

### SECTION 6.4 — THESIS BREAK DEFINITIONS AND TRIPWIRE SYSTEM

Price drops of -25% to -40% are NOT thesis breaks. Thesis breaks are BUSINESS EVENTS only.

**THESIS BREAK DEFINITIONS (per ticker)**

- **VRT:** Hyperscaler capex declines for 2 consecutive quarters OR Book-to-Bill falls < 1.0 for two consecutive quarters OR major hyperscaler officially awards primary cooling contract to competitor.
- **MU:** HBM ASP declines for 2 consecutive quarters AND a forward guidance cut is issued in the same earnings print as, or subsequent to, the second consecutive ASP decline quarter OR NVIDIA switches primary HBM supplier away from MU/SK Hynix OR total company gross margin falls < 20% for 2 consecutive quarters due to legacy DRAM/NAND pricing deterioration.
- **APH:** Total revenue growth (Organic + M&A) falls < 5% YoY for 2 consecutive quarters OR defense/datacom segment combined drops < 45% of total revenue mix.
- **ANET:** MSFT or Meta publicly announce primary switching vendor change away from Arista OR publicly shift > 20% of networking spend to internal "white box" custom solutions within a single fiscal year OR AI/cloud vertical drops < 60% of new bookings for 2 consecutive quarters. NOTE: Margin floor permanently breaches < 60% due to aggressive cloud titan pricing leverage also constitutes a thesis break. Microsoft/Meta formally announcing structurally lower AI networking CapEx also constitutes a thesis break.
- **TSM:** Formal US military repositioning in Taiwan Strait OR sustained Chinese naval quarantine/blockade physically halts wafer shipments for > 30 days OR US government mandates divestment.
- **VST:** Hyperscaler (Meta, MSFT, AMZN) formally cancels or materially reduces a Long-Term Power Purchase Agreement (PPA) OR regulatory/construction delays push contracted nuclear capacity expansion > 18 months behind announced timeline OR new hyperscaler behind-the-meter nuclear PPAs permanently re-price below the $100/MWh baseline combined with two consecutive quarters of hyperscaler energy CapEx guidance cuts. 🟡 YELLOW WATCH TRIGGER (ratified Mar 10, 2026 — INVEST-04): Two consecutive quarters of hyperscaler energy CapEx guidance cuts OR forward baseline power pricing (ERCOT/PJM) dropping > 15% over a 6-month period. ACTION: If triggered, DCA into VST is frozen (Factor 3 score zeroed) and the Victory Protocol soft stop is tightened to 1.25x cost. Yellow Watch does NOT constitute a thesis break — it is a DCA freeze pending resolution.
- **FN:** Lumentum or Nvidia formally shifts primary optical manufacturing contract to a competitor OR Datacom revenue growth drops < 10% YoY for two consecutive quarters OR the industry achieves mass commercialization of Co-Packaged Optics (CPO) that actively cannibalizes FN's high-speed pluggable transceiver volume.
- **ASTS:** N/A. Max pain accepted. Exit trigger is 15% CAP RULE only.
- **ONDS:** Sentrycs loses US Army or DHS anchor contract OR position violates 15% CAP RULE.

**THESIS INTERPRETATION FLAGS**

- **FLAG A — IGNORE CONSENSUS** (applies to: ASTS, ONDS): Maximum pain accepted. High-conviction asymmetric bet. Normal analyst consensus does not trigger a review. Only the position's own tripwires matter.
- **FLAG B — FINANCIALS PRIMARY, CONSENSUS SECONDARY** (applies to: MU, TSM): Standard Core position where financial metrics are primary signal. Tripwire breach triggers mandatory /council trim session before any further capital deployment.
- **FLAG C — GAAP/ADJ FILTER REQUIRED** (applies to: VST): GAAP EPS figures require adjustment filter. Unrealized hedge losses excluded from operational assessment. Use Adj EBITDA as primary metric.
- **NO FLAG** (applies to: VRT, APH, ANET, FN): Standard Core position. Tripwire breach triggers mandatory /council trim session before any further capital deployment.

**TRIPWIRE BREACH PROCEDURE**
1. IO flags breach at session open.
2. Mandatory /council trim [TICKER] session.
3. Council recommendation + Commander decision before any exit.
4. Do not panic-sell without Council session unless VIX >= 25 AND tripwires are simultaneously breached (dual-gate systemic exit).

### SECTION 6.5 — DCA TARGETING MATRIX (CORE POSITIONS ONLY)

Eligible pool: Core positions only PLUS Core Watchtower candidates with /thesis CLEAR. Satellites are excluded from matrix scoring.

**SCORING FACTORS (100 Points Max):**

**FACTOR 1 — UNDERWEIGHT (40 points max)**
⚠️ THESIS DECAY OVERRIDE: If asset is tagged ⚠️ STALE, Factor 1 score = ZERO regardless of underweight calculation. Run /thesis first to unlock.

| Holding vs Target Size | Points |
|------------------------|--------|
| 0 shares held | 40 |
| Held but < 25% of target size | 30 |
| 25–50% of target size | 20 |
| 50–75% of target size | 10 |
| >= 75% of target size | 0 |

**FACTOR 2 — DISTANCE FROM COST (30 points max)**

**ZERO-SHARE RULE (DC-11 — ratified Mar 15, 2026):** For any Core position with 0 shares held, Factor 2 = 0 automatically. Factor 1 already awards maximum urgency (40 pts) for zero-share positions — the underweight signal is fully captured there. A reference price is not required and must not be maintained in portfolio_state for positions with no actual cost basis. No phantom cost figures. No proxy prices.

| % vs Cost | Points |
|-----------|--------|
| Below cost > 20% | 30 |
| Below cost 10–20% | 20 |
| Below cost 0–10% | 15 |
| Above cost 0–10% | 10 |
| Above cost 10–20% | 5 |
| Above cost > 20% | 0 |
| 0 shares held (no cost basis) | 0 |

**FACTOR 3 — THESIS MOMENTUM (20 points max)**

| Signal | Points |
|--------|--------|
| Strong confirmed tailwind: ALL THREE of — earnings beat AND guidance raise AND sector acceleration. If only 1 or 2 present, score as Moderate (10 pts). | 20 |
| Moderate tailwind (thesis intact, sector positive; OR only 1–2 of Strong conditions met) | 10 |
| Neutral (thesis intact, no directional signal) | 5 |
| Headwind OR Yellow Watch active | 0 |

**FACTOR 4 — CONCENTRATION + TREND CHECK (10 points max)**

| Condition | Points |
|-----------|--------|
| Position < 10% of liquid portfolio + uptrend confirmed | 10 |
| Position < 10% of liquid portfolio + neutral trend | 5 |
| Position 10–20% of liquid portfolio | 3 |
| Position > 20% of liquid portfolio | 0 |

**EXECUTION RULES:**
- RULE 1: HIGHEST SCORE WINS.
- RULE 2: TIES GO TO LOWEST SHARE COUNT. If share count also ties, lower cost basis wins. If both have zero shares and no cost basis, apply DC-13 Tiebreaker (see below).
- RULE 3: NO RE-SCORING MID-CYCLE.
- RULE 4: GATE STILL GOVERNS. Section 6.2 applies independently of Matrix score.
- RULE 5: ASTS REMAINDER IS HARDCODED. After primary DCA, ALL remaining ammo goes to ASTS until position reaches fill target, subject to the Zero-Remainder Rule in Section 6.0.

**DC-13 TIEBREAKER FOR ZERO-SHARE TIED WINNERS (DC-13 addendum — ratified Mar 16, 2026):** When two or more Core positions tie on composite score AND both hold zero shares AND fresh capital is being split between them:

- Step 1 — BLACKOUT PROXIMITY: The position with the earlier blackout window receives a minimum 55% of the fresh capital allocation. The later-blackout (or no-blackout) position receives the remainder.
- Step 2 — ADV GATE THRESHOLD (if blackouts within 7 days of each other OR neither has a blackout): The position with the higher ADV gate threshold receives the larger allocation.
- Step 3 — COMMANDER DISCRETIONARY: If steps 1 and 2 do not resolve the tie, Commander decides. Decision must be explicitly logged in Consensus Log before execution.

This tiebreaker applies to fresh capital splits only.

Matrix Winner = highest composite score for the cycle. Winner claims the full ammo allocation for that cycle.

POINTER: Current Matrix scores and winner → portfolio_state (current version) Section 4 Dashboard [local only].

### SECTION 6.6 — THESIS DECAY GATE, WATCHTOWER, AND DEAD HAND CLAUSE

**THESIS DECAY GATE (ratified Mar 10, 2026 — INVEST-06)**

Each Arsenal position carries a THESIS REVIEW DATE.

Stale trigger — either condition fires the flag:
- a) 90 or more days have elapsed since the last /thesis run, OR
- b) New earnings have been published since the last /thesis run.

Effect of STALE status:
- Factor 1 in DCA Matrix = 0 points.
- Position cannot win the Matrix while STALE.
- Position cannot receive new capital while STALE.

Reset: Run /thesis [TICKER]. Thesis Review Date resets to session date.

NOT SET status: Equivalent to STALE for Matrix scoring purposes. All new Arsenal positions start as NOT SET. Run /thesis immediately after any new position enters the Arsenal.

**NOT SET / STALE ESCALATION RULE (DC-06 — ratified Mar 13, 2026)**

- If any Arsenal position carries NOT SET or STALE thesis status for more than 30 consecutive calendar days → IO flags the position with ⚠️ NEGLECT at every session open. NEGLECT does not add new capital restrictions beyond the existing Factor 1 = 0 rule — it is a visibility flag only.
- If NOT SET or STALE status persists for more than 60 consecutive calendar days → /council trim [TICKER] is automatically triggered at the next session open. The forced /council trim is a review mandate, not an exit mandate. Council assesses whether the position retains Arsenal eligibility.
  - If Council confirms thesis intact → position remains. NEGLECT flag clears on /thesis completion.
  - If Council identifies thesis deterioration → standard tripwire breach procedure applies (Section 6.4).

CLOCK TRACKS ESCALATION: Clock member is responsible for flagging NEGLECT onset date and 60-day forced trim trigger date at each session open.

**WATCHTOWER GRADUATION RULE**

Any ticker in Tier 2 (The Watchtower) must have a defined quantitative BUY TRIGGER and a hard REMOVAL DATE.
- TRIGGER HIT: Promote to Tier 1. Promotion is a STATUS CHANGE, not an execution order. Section 6.2 Gate governs actual entry. If trigger hit during VIX freeze: log promotion, execute on first session Gate fully passes. Removal date PAUSES from trigger date.
- REMOVAL DATE: If trigger not hit by removal date, discharge the ticker. Log in Changelog.

**NEW TICKER WATCHTOWER STANDARD (Amendment F — Mar 10, 2026)**

A ticker may be added to the Watchtower only after:
1. Passing all 4 Alpha Filters (Rules 1–4 of Section 5)
2. /thesis CLEAR verdict
3. Defined BUY TRIGGER and REMOVAL DATE
4. IO vault overlap check confirms non-material overlap
5. Council votes to add (6/11 or Triad vote per Commander discretion)

**DEAD HAND CLAUSE (ratified COMET-03, Mar 12, 2026)**

Trigger condition: No /review command has been executed and logged in portfolio_state for 120 or more consecutive calendar days.

Effect: A mandatory trim trigger activates on ALL Core Arsenal positions simultaneously. No position is exempt. No partial exemption.

Purpose: Forces Commander re-engagement with the full portfolio. Prevents thesis drift through inactivity.

Resolution: Run /review. Confirm theses are current. Deactivate the trigger. /review resets the 120-day clock.

Override: NONE. The Dead Hand Clause is unconditional.

### SECTION 6.7 — SATELLITE DEPLOYMENT RULES

Satellites operate entirely outside the Core Matrix scoring system.

Allocation cap: Maximum 15% of total liquid portfolio value per individual Satellite position. Enforced at execution. Checked via /audit.

Funding rules:
- Option 1: Explicit Commander override to divert Core ammo pool (Missed Entry Protocol override — logged in Consensus Log).
- Option 2: Fresh Tier 1 capital injection designated specifically for the Satellite. Cannot commingle with Core ammo.

Entry sequence:
1. /thesis pass (Alpha Filters 1-4 confirmed).
2. Watchtower graduation (Tier 2 entry logged in portfolio_state).
3. Trigger confirmation (Section 6.2 gate passed).
4. 15% cap check via /audit.
5. Commander authorization for funding source.
6. Execution.

**SATELLITE PRIORITY SCORING TIER (DC-13 — ratified Mar 16, 2026)**

SCOPE: Watchtower Satellites only. Does NOT apply to Core Watchtower candidates or Arsenal Satellites already held.

PURPOSE: Determines deployment priority order among competing Satellite candidates when fresh capital is available.

**SATELLITE PRIORITY SCORE (30 points max):**

FACTOR S1 — EXECUTION URGENCY (10 points max)

| Condition | Points |
|-----------|--------|
| Trigger already hit + expiry window ≤14 days | 10 |
| Trigger already hit + expiry >14 days OR no expiry | 7 |
| Blackout opens ≤21 days (not yet triggered) | 5 |
| Blackout opens 21–45 days (not yet triggered) | 3 |
| No near-term time pressure | 0 |

FACTOR S2 — THESIS CONVICTION DELTA (10 points max)

| Condition | Points |
|-----------|--------|
| Major structural re-rating event since Watchtower entry | 10 |
| Thesis intact + incremental positive catalysts since entry | 6 |
| Thesis intact, no material change since entry | 3 |
| Active ⚠️ WATCH flag (Israel, China, operational disruption) | 1 |

FACTOR S3 — CONCENTRATION HEADROOM (10 points max)

| Condition | Points |
|-----------|--------|
| 0 shares held — full 15% headroom available | 10 |
| Partial position held — >50% of 15% cap remaining | 6 |
| Partial position held — <50% of 15% cap remaining | 3 |
| At or near 15% cap | 0 |

SCORING RULES:
- Run at /dca time alongside Core Matrix.
- Highest Satellite Priority Score = funded first from available Satellite capital.
- Ties broken by S1 (urgency) first, then Commander discretionary. Log in Consensus Log.
- Score logged in portfolio_state Section 4 Dashboard.

**AGGREGATE SATELLITE SLEEVE LIMIT:** Maximum 40% of total liquid portfolio may be held in Satellite positions simultaneously. IO checks aggregate Satellite exposure at every /audit. If deployment would breach 40%: IO flags. Commander must explicitly override or defer before executing. This limit does not replace the per-position 15% cap — both apply independently.

### SECTION 6.8 — SNIPER ROE MATRIX

**THE CORE (VST, VRT, MU, APH, ANET, TSM, FN):** Goal = Level 2 (Growth). Stop Loss = THESIS BREAK (Section 6.4) and VICTORY PROTOCOLS. Trim only if a single asset > 40% of total liquid portfolio.

**THE SATELLITE (ASTS, ONDS — and any future Arsenal Satellites):** Goal = Level 2 (Acceleration). Stop Loss = None (-100% max pain). Max allocation: Must not exceed 15% of total liquid portfolio. CAP ENFORCEMENT: If a spike drives Satellite above 15% cap, no new capital may enter until position retraces below cap via price action. Do not force-trim on a spike unless concentration breach persists for > 30 days. SATELLITE EXPANSION: /thesis CLEAR required; discretionary capital only; Gate applies; CAP CHECK must pass.

**ARSENAL SATELLITE EXPANSION GATE (DC-13b — ratified Mar 16, 2026):** Once an Arsenal Satellite reaches its established fill target, any further capital expansion beyond that fill target requires ALL THREE of the following conditions before execution:
- a) Fresh /thesis CLEAR for that ticker — must have been run within the past 90 days.
- b) /audit confirming 15% per-position cap headroom exists after the proposed expansion.
- c) Explicit Commander authorization — stated in session and logged in Consensus Log before execution.

**THE SHIELD:** Goal = Level 1 (Liquidity). Stop = Yield < 5% OR NPLs > 3.5% (TISCO/ADVANC). Condo: net equity tracked separately.

**THE VAULT (RMF / SSF / SSFA / SSFE):** Goal = Level 3 (Freedom). Locked 20 years. No cash redemption. E-CLASS EXCEPTION: E-Class wrapper funds are LIQUID and SWITCHABLE. Count as TIER 2 (80% haircut) toward Level 1 target. All E-Class switches governed by Section 6.9.

### SECTION 6.9 — VAULT REBALANCING PROTOCOL

**SWITCH TRIGGERS (any ONE sufficient):** Thesis break at master ETF level | Sustained underperformance vs VOO > 15pp over 12 months | Extreme vault concentration > 50% | Pending earnings binary event blackout (no switch 7 days before/after key earnings event).

POINTER: Vault fund details and E-Class positions → vault_registry (current version) [local only].

**/VAULT_REVIEW — ANNUAL REVIEW PROCEDURE**

Run annually each December, or when a switch trigger fires, or when vault total exceeds ฿500,000. Activated by /vault_review command. Do NOT run in response to short-term volatility.

- **STEP 1 — VAULT SNAPSHOT:** Pull current vault state from vault_registry. Record total cost, current value, return, full exposure map, E-Class liquid total, and concentration flag.
- **STEP 2 — BENCHMARK YTD PERFORMANCE:** Fetch YTD price data for SOXX, VOO, QQQ, EWY, GLD via MCP. Calculate YTD % for each. Compute delta vs VOO baseline. Flag any benchmark >15pp below VOO over 12 months.
- **STEP 3 — TRIGGER ASSESSMENT:** Check all 4 switch triggers. If zero triggers: HOLD ALL. Log in Consensus Log. Session complete. If one or more triggers: proceed to Step 4.
- **STEP 4 — SWITCH EVALUATION:** For each triggered fund: confirm trigger → thesis assessment (structural vs cyclical?) → target fund selection → timing gate check → switch recommendation (EXECUTE / DEFER / ABORT with reason).
- **STEP 5 — DECISION AND LOGGING:** Commander decides. Execute or log hold. Update vault_registry and portfolio_state Consensus Log.

### SECTION 6.10 — BLACKOUT PROTOCOL (DC-05 — ratified Mar 13, 2026)

**DEFINITION:** A blackout window is any period during which insider-trading restrictions, earnings proximity rules, or Commander-designated trading halts prohibit new order placement for a specific ticker.

**GENERAL RULE:** No new buy or sell orders may be placed for a blacked-out ticker after the blackout window opens.

**PRE-PLACED ORDER CARVE-OUT:** Fill orders submitted to the broker before a blackout window opens remain valid for execution during the blackout period. The blackout rule prohibits new order placement only — it does not cancel or invalidate orders already placed.

**IO VERIFICATION REQUIREMENT:** Before granting carve-out status to any pending order, IO must confirm that the order submission timestamp predates the blackout window open. If submission timestamp cannot be confirmed, the order is treated as post-blackout and subject to the general blackout rule.

**LOGGING:** All carve-out executions are logged in the Consensus Log with the tag: ⚠️ BLACKOUT CARVE-OUT — PRE-PLACED.

---

## SECTION 7 — VICTORY PROTOCOL

### VICTORY PROTOCOLS — POSITION REVIEW (2x Rule)

When a Core position doubles from cost (2x), conduct a Position Review:
1. Does this single position now exceed 40% of total liquid portfolio? → Trim to 30%.
2. Is the original thesis still intact per Section 6.4? → If NO: exit.
3. Has the stock's forward P/E expanded above 2x its historical average? (EXEMPT: MU and TSM) → If YES: trim 25%, set soft review stop at 1.5x cost.
4. If all NO/PASS: hold. Set universal soft review stop at 1.5x cost.

### CAPITAL RECAPTURE PROTOCOL (The 3x Rule)

**SUB-$500 WAIVER:** Original capital below $500 → waive sale, apply Trailing Shield.

For >= $500: sell enough shares to recover original capital deployed.

**THE TRAILING SHIELD:** 3x → stop@2x | 4x → stop@3x | 5x → stop@4x. Never moves down.

APPLIES TO: VST, VRT, MU, APH, ANET, TSM, FN.
DOES NOT APPLY TO: ASTS, ONDS.

### EXIT CONDITIONS

**Full Exit Triggers (per position):**
- Thesis breaks (tripwire breached + /council trim vote confirms exit).
- Systemic drawdown (VIX >= 25 + multiple tripwires breached simultaneously).
- Dead Hand Clause activated + /review confirms thesis invalidation.

**Partial Exit Triggers:**
- Concentration breach (position exceeds 25% of total liquid portfolio — trim to 20%).
- Rebalancing directive from /council rebalance session.

**Hold Through Volatility:** Short-term price swings do not trigger exits unless tripwires are breached. VIX spikes alone do not force selling unless paired with thesis deterioration.

### PROFIT TAKING

No systematic profit-taking rules. Exits are thesis-driven, not price-driven. If thesis remains intact, position holds regardless of gains.

### REBALANCING

Quarterly /review includes rebalancing check. Concentration limits: Single position max: 25% of total liquid portfolio (trim trigger at 25%, target 20%). Sector max: 40% of total liquid portfolio.

POINTER: /review protocol → doctrine_ops (current version) Section 0.4.

---

## THE CAMPAIGN MAP (THE VISION)

**ACTIVE LEVEL: LEVEL 1 — All decisions optimize for this target first.**

**LEVEL 1: "THE BAHT MILLIONAIRE"**
Target: 1,000,000 THB LIQUID

LIQUID DEFINITION: Assets count toward the target ONLY if convertible to THB within 5 business days.

| Tier | Assets | Value Haircut |
|------|--------|---------------|
| TIER 1 — HARD CASH | THB Bank Balances and PromptPay accounts | 100% |
| TIER 2 — MARKET LIQUID | US Equities, Thai defensive stocks, Standard Mutual Funds (Non-SSF/RMF/SSFA/SSFE), E-CLASS VAULT FUNDS | 80% |
| TIER 3 — ILLIQUID | SSF wrapper funds, RMF wrapper funds, Condo net equity | 0% — Do not count toward Level 1 target |

**Level 1 Liquid Formula:** (Tier 1 Cash) + (Tier 2 Assets × 0.80)

Sub-milestones (M-1):
- M-1a: ฿100K — Arsenal established (3+ positions)
- M-1b: ฿200K — Watchtower active (3+ slots)
- M-1d: ฿500K — Mid-Level review, Vault review
- M-1e: ฿800K — Pre-Flight: strategy + vault rebalance (THE VANGUARD PROTOCOL — INVEST-10)
- **M-1 Complete: ฿1,000K** — Victory Session, Level 2 declaration

POINTER: Current Level 1 progress → portfolio_state (current version) Section 4 [local only].

**THE VANGUARD PROTOCOL (ratified Mar 10, 2026 — INVEST-10):** When /audit confirms Total Liquid Portfolio >= ฿800,000 (gap <= ฿200,000), the IO activates 🟡 LEVEL 2 PRE-FLIGHT status. The subsequent session must be dedicated to drafting the Level 2 DCA Matrix variant and Shield reassessment. Until this trigger is hit, Level 1 strict capital discipline remains absolute. Level 2 deployment rules are NOT active until Pre-Flight session completes.

**LEVEL 2: "THE MUNGER MOMENT"**
Target: $100k USD Total Portfolio. Status: Unlocks after Level 1.

**LEVEL 3: "THE FREEDOM FUND"**
Target: $500k USD | 4% Rule = $20k/year passive. Status: LONG-RANGE PRIMARY GOAL (20+ year horizon).

---

*END OF DOCTRINE CORE v1.12.1*
