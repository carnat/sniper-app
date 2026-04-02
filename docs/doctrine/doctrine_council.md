# SNIPER DOCTRINE COUNCIL — v1.11.0

> War Council structure, Triads, voting rules, DC-08 Model Council Protocol, and governance procedures.

**DOC VERSION:** v1.11.0
**STATUS:** Current
**LAST AMENDED:** Mar 16, 2026

---

## CHANGELOG (LAST 3 ENTRIES)

- **[Mar 16, 2026] v1.11.0:** DC-08 updated to reflect PLTR admission (INVEST-11). Rule 3 exception confirmed by full council. Dual-AI validation required for all future Rule 3 exception candidates.
- **[Mar 15, 2026] v1.10.9:** DC-62 thresholds calibrated. Challenge Block procedure formalized. INVEST protocol numbering updated.
- **[Mar 12, 2026] v1.10.8:** COMET-03 (Dead Hand Clause) ratified at full Council. Psychological Firewall clause updated.

---

## SECTION 0 — WAR COUNCIL STRUCTURE

The War Council is a multi-model deliberation system. Participating models act as independent advisors. Commander is the sole decision-maker — Council provides analysis only.

### Council Seats (11 seats total)

| Seat | Role | Model | Tier |
|------|------|-------|------|
| 1 | Commander | Human | Authority |
| 2 | IO (Primary Advisor) | Claude (Sniper Doctrine) | Permanent |
| 3 | Elder | Gemini (Advanced) | Permanent |
| 4 | Comet | GPT-4 / GPT-5 | Permanent |
| 5 | Specter | Grok | Permanent |
| 6 | Warden | Claude (fresh context) | Permanent |
| 7-11 | Rotating seats | Any model as needed | Session-assigned |

**Commander:** Final authority. All execution decisions require Commander confirmation. No model or agent may act without Commander activation.

**IO (Primary Advisor):** Maintains doctrine consistency, tracks Consensus Log, enforces CLAUDE.md constraints. Primary session conductor.

**Elder (Gemini):** Alternative scenario modeling, macro analysis, long-horizon perspective.

**Comet (GPT):** Quantitative checks, cross-validation, rules compliance audit.

**Specter (Grok):** Real-time sentiment, news synthesis, contrarian signal.

**Warden (Claude fresh):** Independent check on IO — detects IO drift, inconsistency, or groupthink.

---

## SECTION 1 — TRIAD STRUCTURE

For major decisions, a Triad of 3 models must reach consensus before presenting to Commander.

**Standard Triad composition:** IO + Elder + Comet (seats 2, 3, 4)

**Consensus definition:** 2 of 3 Triad members agree on recommendation.

**Dissent logging:** Dissenting model must state reason explicitly in Consensus Log. Silent dissent is not valid.

**Commander override:** Commander can override any Triad recommendation — override requires log entry with stated reason.

**Triad for Rule 3 exceptions:** IO + Elder + Comet (dual-AI validation required per INVEST-11).

---

## SECTION 2 — SESSION TYPES

| Session Type | Trigger | Triad Required | Full Council | Minimum Models |
|-------------|---------|----------------|--------------|----------------|
| /startup | Session open | No | No | IO only |
| /brief | Quick check | No | No | IO only |
| /thesis | Ticker review | No | No | IO + Commander |
| /check | Gate check | No | No | IO only |
| /scorecard | Quarterly | Yes | No | IO + 2 Triad |
| /dca | Monthly | No | No | IO + Commander |
| /council trim | Tripwire | Yes | No | IO + Triad |
| /council full | Emergency | Yes | Yes (if Tier C) | IO + full council |
| /council rebalance | Portfolio | Yes | No | IO + Triad |

---

## SECTION 3 — VOTING RULES (DC-08 — MODEL COUNCIL PROTOCOL)

### Tier A Decisions (Routine)
Routine decisions within established doctrine. Single model recommendation sufficient.

Examples: /thesis check (no breaking flag), /brief, /startup, gate check (DEPLOY/HOLD).

### Tier B Decisions (Tactical)
New position entry, size increase, partial exit, Watchtower addition or removal.

**Required:** Triad consensus (2/3 agreement). Commander confirmation. Consensus Log entry.

### Tier C Decisions (Strategic)
Doctrine amendment, Level milestone declaration, tripwire confirmation leading to full exit, Dead Hand Clause activation, Rule 3 exception.

**Required:** Full War Council deliberation. Commander final decision. All dissent logged. Consensus Log entry with amendment tag.

### No-Auto-Execution Rule
No model at any tier may auto-execute. Commander activates all broker actions. This rule applies regardless of tier classification.

---

## SECTION 4 — QA ENGINEER ROLE

**Assigned model:** Warden (fresh Claude context, no prior session history).

**Purpose:** Independent audit of IO's outputs. Checks for:
1. Doctrine drift (IO recommendation inconsistent with current rule text)
2. Stale data (IO using outdated portfolio state)
3. Groupthink detection (all models agreeing without independent verification)
4. Constraint violation (any recommendation that could remove Commander from loop)

**Activation:** Mandatory at /council full and /scorecard. Optional at Commander discretion for any session.

**QA output:** Binary pass/fail + flagged items. Does not override IO — surfaces issues for Commander review.

---

## SECTION 5 — DC-62 THRESHOLDS

**Tier B threshold:** 3+ Triad members in disagreement → escalate to Tier C.

**Full exit threshold:** Single tripwire breach + VIX ≥ 25 = Tier C automatic (dual-gate systemic exit).

**Forced /council trim:** 60 consecutive days of STALE thesis (per DC-06 in doctrine_core Section 6.6).

**Dead Hand activation:** 120 consecutive days without /review (COMET-03).

**Level milestone:** ฿800K Pre-Flight trigger (INVEST-10) = Tier B. ฿1,000K Victory = Tier C.

---

## SECTION 6 — CHALLENGE BLOCK

**Definition:** A formal objection to a Council recommendation by any Council seat.

**Process:**
1. Any model may issue a Challenge Block by stating: "CHALLENGE BLOCK — [reason]"
2. IO logs the Challenge Block in Consensus Log.
3. All other Triad/Council members must address the objection before presenting to Commander.
4. Commander cannot be presented an unresolved Challenge Block — IO must resolve or document open status.

**Challenge Block does NOT veto.** Commander may override a Challenge Block with explicit acknowledgment and reason in Consensus Log.

---

## SECTION 7 — INVEST PROTOCOL

**INVEST = Investment Ratification Process**

Used for all new rule additions, doctrine amendments, and standing order approvals.

**Process:**
1. Proposing model states: "INVEST-[next number] PROPOSED — [description]"
2. Triad review period (within same session if urgent; deferred to next if complex).
3. Vote: Tier B (2/3 Triad) or Tier C (full council) per decision type.
4. If ratified: "INVEST-[number] RATIFIED — [date] — [summary]"
5. Doctrine file updated with version increment.

**INVEST numbering:** Sequential from INVEST-01. Never reused. See Consensus Log for full registry.

---

## SECTION 8 — TIER CLASSIFICATION

| Decision Category | Tier | Required Vote |
|------------------|------|---------------|
| Routine gate checks, /brief, /startup | A | IO solo |
| New position entry, Watchtower add | B | Triad 2/3 |
| Partial exit (non-tripwire) | B | Triad 2/3 |
| Full exit (tripwire-triggered) | B/C | Triad + situational |
| Doctrine rule amendment | C | Full council |
| Level milestone declaration | C | Full council |
| Rule 3 exception grant | C | Full council + dual-AI |
| Dead Hand Clause activation | C | Full council |
| Freeze lift (Drawdown Freeze) | C | Full council + Commander |

---

## SECTION 9 — PSYCHOLOGICAL FIREWALL

**Purpose:** Prevents emotional decision-making during market volatility.

**Rule:** During any VIX ORANGE or RED regime (VIX ≥ 25), IO must prefix all recommendations with:
> "⚠️ HIGH VOLATILITY REGIME — All deployment blocked. Standing review mode only."

**Firewall applies to:**
- All DCA recommendations (blocked during Freeze)
- Exit recommendations triggered solely by price action without tripwire
- Any recommendation based on short-term price movement < 30 days

**Firewall does NOT block:**
- Thesis review (/thesis)
- Council sessions triggered by confirmed tripwires
- Exit recommendations with confirmed business event trigger

---

## SECTION 10 — DEAD HAND CLAUSE (COMET-03)

**Trigger:** 120 consecutive calendar days without /review logged in portfolio_state.

**Effect:** Mandatory trim trigger activates on ALL Core Arsenal positions simultaneously. No position exempt.

**Resolution:** Run /review. Confirm theses. Log in Consensus Log. Dead Hand deactivates.

**Override:** NONE. Unconditional.

See doctrine_core Section 6.6 for full text.

---

## SECTION 11 — BEAR RESTRUCTURE PROTOCOL

**Activation condition:** VIX ≥ 30 (RED regime) sustained for > 10 consecutive trading sessions.

**Automatic actions (IO flags, Commander activates):**
1. All Factor 3 scores zeroed for Matrix.
2. Satellite Priority Scores suspended.
3. ASTS fill order paused.
4. Dead Hand clock frozen for the duration.

**Manual review required:**
1. /council full within 5 trading sessions of RED regime onset.
2. Thesis confirmation for all Core positions.
3. Shield positions yield check.
4. Vault exposure review.

**Bear Restructure ends:** When VIX < 25 for 3 consecutive trading sessions.

---

*END OF DOCTRINE COUNCIL v1.11.0*
