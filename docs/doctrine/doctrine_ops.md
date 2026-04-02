# SNIPER DOCTRINE OPS — v1.15.1

> Operational rules, session protocols, slash commands, and data source procedures.

**DOC VERSION:** v1.15.1
**STATUS:** Current
**LAST AMENDED:** Mar 16, 2026

---

## CHANGELOG (LAST 3 ENTRIES)

- **[Mar 16, 2026] v1.15.1:** PLTR added to Core Watchtower roster (INVEST-11, Rule 3 exception confirmed). BWXT reclassified as Core (previously Satellite). DC-95 DRIP protocol updated to reflect new BWXT classification.
- **[Mar 15, 2026] v1.15.0:** AAOI permanently removed from Watchtower and blacklisted. Watchtower capacity updated. DC-95 DRIP rules clarified for switched fund wrappers.
- **[Mar 14, 2026] v1.14.9:** Rule 12 added — Mandatory Consensus Log sync at session close. IO verification requirement for all /close commands.

---

## SECTION 0 — OPERATIONAL RULES

### RULE 1 — SESSION INTEGRITY
Every session must open with a /startup or /brief. No DCA decisions without a gate check.

### RULE 2 — SINGLE SOURCE OF TRUTH
portfolio_state is the only authoritative source for portfolio positions, ammo, and HWM. All calculations derive from it.

### RULE 3 — NO SILENT HOLDS
Every deferred decision must be explicitly logged as "defer" with a date in the Consensus Log. Silence is not a decision.

### RULE 4 — VERSION CONTROL
Every doctrine amendment increments the version. No undocumented changes. POINTER format used for cross-references between doctrine files.

### RULE 5 — COMMANDER-IN-LOOP (CARDINAL RULE)
No model, tool, automation, or agent may place, queue, simulate, or proxy a broker order of any kind. All execution is Commander-initiated. This rule cannot be amended, overridden, or suspended.

### RULE 6 — IO CONSISTENCY GATE
If IO produces a recommendation that contradicts a standing rule in doctrine_core, IO must flag the contradiction explicitly before presenting the recommendation. Silent contradictions are prohibited.

### RULE 7 — DATA FRESHNESS
Price data older than 1 business day is marked STALE in any gate check output. Stale data does not block a session — it adds a ⚠️ STALE DATA caveat to all outputs.

### RULE 8 — THESIS AGE CLOCK
- 0–30 days: CURRENT (green)
- 31–60 days: AGING (yellow, ⚠️ flag at next session)
- 61–89 days: NEGLECT (orange, NEGLECT flag active per DC-06)
- 90+ days: STALE (red, Factor 1 = 0, DCA blocked)

### RULE 9 — COUNCIL LOG VERSIONING
Consensus Log entries are sequential within a session. Session ID = date (YYYY-MM-DD). Cross-session references use the date prefix.

### RULE 10 — CROSS-MODEL VALIDATION
Any recommendation tagged as Tier B or Tier C (per doctrine_council) requires at minimum one cross-model validation before Commander presentation. Cross-model = a different AI model than the one originating the recommendation.

### RULE 11 — EARNINGS PROXIMITY ALERT
IO automatically flags earnings proximity at session open if any Arsenal or Watchtower ticker has a scheduled earnings date within 30 days. Blackout windows are pre-loaded and cannot be manually extended without /council approval.

### RULE 12 — MANDATORY CLOSE SYNC (DC-62 — ratified Mar 14, 2026)
All /close commands must include an explicit Consensus Log sync confirmation. IO verifies that all open decisions from the session have been logged or deferred before closing. Unclosed decisions block /close until resolved.

---

## SECTION 0.2 — DATA SOURCE PROTOCOL

**Price data:** yfinance (US equities) — 20-day ADV, 52-week high/low, SMA50, SMA200, daily close.

**Macro data:**
- VIX (^VIX): primary regime gate. Source: yfinance.
- THB=X: FX rate gate. Source: yfinance.
- SPY SMA200: used for SPY above/below 200DMA flag.

**Vault NAV data:** Commander-supplied at each /close or /vault_review session. Not automated — Commander screenshots or manually inputs NAV from broker platform.

**Earnings dates:** IO retrieves from yfinance or public calendar. Commander confirms final date at session.

**News data:** Fetched via yfinance news API (max 3 headlines per ticker per session).

---

## SECTION 0.3 — COUNCIL INTEGRATION NOTES

**Triad composition for each session type:**
- /startup: Auto-brief (no Triad required)
- /thesis: IO solo + Commander review
- /scorecard: Full Triad (3 models) + Commander
- /council: Full War Council (11 seats) + Commander
- /brief: IO solo

**INTEGRITY RULE (cost vs price interpretation):**
Never confuse a cost-basis problem for a thesis problem. % vs Cost reflects ENTRY TIMING, not asset health. Always run /review to see master ETF YTD performance separately from individual position cost basis.

**/REVIEW YTD SWEEP PROTOCOL:**
Run at each quarterly scorecard session and after any market regime change (VIX crosses 22, 25, or 30 thresholds). Fetches current YTD performance for all benchmark ETFs (SOXX, VOO, QQQ, EWY, GLD). Generates delta vs VOO. Flags any >15pp underperformance.

---

## SECTION 0.4 — WATCHTOWER ROSTER MANAGEMENT (DC-40)

**Capacity:**
- Core tier: Up to 12 concurrent positions (S1/S2/S3 sub-tiers)
- Satellite tier: Up to 3 concurrent positions (SAT-1/SAT-2/SAT-3 sub-tiers)
- Total capacity: 15 positions

**Tier Definitions:**
- S1 (Core Tier 1): Highest conviction — trigger imminent or recurring DCA active
- S2 (Core Tier 2): High conviction — waiting for gate clear
- S3 (Core Tier 3): Monitoring — thesis confirmed, queued for next scorecard
- SAT-1/2/3: Satellite tiers — deploy from Satellite capital only (DC-13 score ≥25 req'd)

**Current roster:** See portfolio_state (current version) [local only].

**Admission criteria:**
1. Pass all Alpha Filters (Rules 1-4 of doctrine_core Section 5)
2. /thesis CLEAR verdict
3. Defined BUY TRIGGER and REMOVAL DATE (max 6 months unless extended by /council)
4. IO vault overlap check confirms non-material overlap
5. Council vote to add (6/11 or Triad vote per Commander discretion)

**Removal triggers (DC-40):**
1. Removal date reached with no trigger → free slot, no execution
2. Thesis break confirmed → immediate removal + Consensus Log
3. Core graduation → promoted to Arsenal, slot freed
4. Satellite deploy complete → hold as Arsenal position, slot freed
5. Alpha Filter breach (China >20%, permanent ban) → immediate removal

**Blackout windows:** No new DCA during earnings blackout (±30d from earnings date). Pre-placed orders exempt per DC-05.

---

## SECTION 0.5 — TICKER MAP (CURRENT)

**Arsenal Core:** VRT, ASTS, VST, MU, APH, ANET, TSM, ONDS, FN, COHR

**Watchtower Core (12):** TSEM, BWXT, MOD, NBIS, FORM, ENTG, ONTO, LITE, QRVO, PLTR, KTOS, SKYT

**Watchtower Satellite (3):** RKLB, SATL, PL

**Sector ETFs:** SOXX, ITA, XLU

**Benchmarks:** VOO, QQQ, EWY, GLD, SPY

**Macro:** ^VIX, THB=X

POINTER: Arsenal actual positions and shares → portfolio_state (current version) [local only].

---

## SECTION 1 — SLASH COMMANDS

### /startup
**Purpose:** Opening session read. Mandatory at start of every session.
**IO produces:**
1. VIX regime status (GREEN/YELLOW/ORANGE/RED)
2. SPY vs 200DMA status
3. THB=X zone (A/B/C)
4. Active Standing Orders with expiry status
5. Thesis age for each Arsenal position (flag NEGLECT/STALE)
6. Active ⚠️ flags (China Watch, Israel Watch, Yellow Watch, Drawdown Freeze)
7. HWM status and drawdown calculation
8. Next earnings dates within 30 days

### /brief
**Purpose:** Condensed situational awareness. Gate status + top priority only.
**IO produces:**
1. VIX + regime (1 line)
2. Gate OPEN/CLOSED verdict (1 line)
3. Highest Matrix score winner (1 line)
4. Any active STOP flags (1 line each)

### /check [TICKER]
**Purpose:** Single-ticker gate check.
**IO produces:**
1. Q1 VIX gate pass/fail
2. Q2 volume confirmation (last session vs ADV)
3. Q3 extension check (% above reference price)
4. Q4 blackout status
5. Gate verdict: DEPLOY / HOLD / BLOCKED

### /thesis [TICKER]
**Purpose:** Full thesis review and Alpha Filter check for one ticker.
**IO produces:**
1. Alpha Filter 1-4 pass/fail
2. China Watch / Israel Watch flag status
3. Thesis statement (2-sentence summary)
4. Tripwire status (per Section 6.4 of doctrine_core)
5. Thesis verdict: CLEAR / FLAG / BREAK
6. Sets/resets Thesis Review Date on completion

### /scorecard
**Purpose:** Quarterly Matrix scoring session. Full Triad required.
**IO produces:**
1. DCA Targeting Matrix for all Core positions (F1-F4 scores)
2. Satellite Priority Scores for all Watchtower Satellites (S1-S3 scores)
3. Matrix Winner declaration
4. Satellite Priority Order
5. Recommended ammo allocation for cycle

### /dca
**Purpose:** Monthly DCA execution preparation.
**IO produces:**
1. Matrix Winner for cycle (from last /scorecard or current calculation)
2. Q1-Q4 gate status
3. FX Zone
4. Recommended deployment amount
5. ASTS fill order status (subject to Zero-Remainder Rule)
6. All pre-execution checklist items (threshold, blackout, ADV)

### /audit
**Purpose:** Deep-dive compliance audit.
**IO produces:**
1. All Alpha Filter checks for current Arsenal
2. Concentration check (per-position vs 25% cap, Satellite vs 15% cap, aggregate Satellite vs 40%)
3. HWM update and drawdown calculation
4. Neglect clock status for all positions
5. Active ⚠️ flags
6. Dead Hand Clause clock (days since last /review)

### /council [type]
**Purpose:** War Council deliberation session.
**Types:** trim [TICKER] | full | rebalance | tripwire [TICKER]
**IO produces:**
1. Trigger condition summary
2. Council recommendation (requires Triad consensus for Tier B, full council for Tier C)
3. Options presented to Commander (hold / reduce / exit / defer)
4. Decision logged to Consensus Log

### /review
**Purpose:** Portfolio-wide thesis refresh + benchmark YTD sweep. Resets Dead Hand Clause clock.
**IO produces:**
1. Thesis status for all positions
2. Benchmark YTD performance vs VOO
3. Vault exposure map delta
4. HWM update
5. Rebalancing check (concentration limits)
6. Dead Hand clock reset logged

### /vault_review
**Purpose:** Annual vault rebalancing review. Runs 5-step procedure (doctrine_core Section 6.9).

### /close
**Purpose:** Session close. Requires Consensus Log sync (Rule 12 / DC-62).
**IO produces:**
1. Session summary (decisions made, deferred items)
2. Open item confirmation (all decisions logged or explicitly deferred)
3. Next session trigger (scheduled date or event)

---

## SECTION 2 — TA RULES (TECHNICAL ANALYSIS GATES)

### TA-1 — SMA200 REGIME GATE
- SPY above SMA200: OFFENSIVE mode (full deployment eligible)
- SPY below SMA200: DEFENSIVE mode (deployment restricted to S1 Core positions only; no new Satellite entries)

### TA-2 — SMA50 CONFIRMATION
- Target above SMA50: MOMENTUM CONFIRMED (Factor 3 bonus eligible)
- Target below SMA50 but above SMA200: RECOVERING (no Factor 3 momentum bonus)
- Target below SMA200: DOWNTREND (Factor 3 = 0 points)

### TA-3 — VOLUME GATE (per doctrine_core Rule 4)
- Last session volume ≥ 1.5× 20-day ADV: VOLUME CONFIRMED
- Below threshold: HOLD — do not execute. Wait for next qualifying session.

---

## SECTION 3 — DC-95 DRIP PROTOCOL

**SCOPE:** Applies to E-Class vault funds (Tier 2 liquid wrappers) only. Does NOT apply to locked funds (SSF/RMF/SSFA/SSFE).

**RULE:** Any dividend or distribution received from an E-Class fund is automatically reinvested (DRIP) unless Commander issues explicit redirect at the session following distribution date.

**REINVESTMENT PRIORITY:**
1. Same fund (DRIP direct) — default behavior
2. Redirect to different E-Class fund only if T1 switch trigger fires (Section 6.9 of doctrine_core)
3. Redirect to Arsenal ammo only with explicit /council approval + Commander override

**DRIP LOGGING:** All DRIP events are logged in the Consensus Log with the tag: DRIP — [FUND] — [AMOUNT].

**DRIP does not change locked fund classification.** If a locked fund pays a distribution that is reinvested in the same wrapper, the reinvested units remain Tier 3 (locked). E-Class to E-Class DRIP maintains Tier 2 status.

---

*END OF DOCTRINE OPS v1.15.1*
