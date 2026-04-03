# SNIPER DOCTRINE CORE MATRIX — v1.0.0-BETA.1

> Extended matrix scoring methodology: F1–F5 factors, DC-40 conviction tiers, DC-43 regime scoring.
> Extends doctrine_core Section 6.5.

**DOC VERSION:** v1.0.0-BETA.1
**STATUS:** BETA — Pending full council ratification
**LAST AMENDED:** Mar 16, 2026

---

## CHANGELOG (LAST 3 ENTRIES)

- **[Mar 16, 2026] v1.0.0-BETA.1:** Initial extraction from doctrine_core Section 6.5 + Watchtower Scorecard quarterly protocol. DC-43 regime scoring added. DC-40 conviction tiers formalized.
- **[Mar 16, 2026] v0.9.9:** DC-13 Satellite Priority Scoring Tier added. 30-point Satellite score system documented.
- **[Mar 15, 2026] v0.9.8:** DC-11 Zero-Share Rule added. Reference prices for zero-share positions prohibited.

---

## SECTION 0 — MATRIX OVERVIEW

The DCA Targeting Matrix is the primary mechanism for selecting which Core Arsenal or Core Watchtower position receives the monthly ammo allocation.

**Two separate scoring systems exist:**
1. **Core Matrix (100 points max):** Determines which Core position is the Matrix Winner for the deployment cycle. Applies to Core Arsenal + Core Watchtower candidates with /thesis CLEAR.
2. **Satellite Priority Score (30 points max):** Determines deployment priority among competing Watchtower Satellite candidates. Applied separately — Satellites do NOT compete against Core positions.

---

## SECTION 1 — CORE DCA TARGETING MATRIX

### FACTOR 1 — UNDERWEIGHT (40 points max)

⚠️ THESIS DECAY OVERRIDE: If asset is tagged ⚠️ STALE or ⚠️ NOT SET, Factor 1 score = ZERO regardless of underweight calculation. Run /thesis first to unlock.

| Holding vs Target Size | Points |
|------------------------|--------|
| 0 shares held | 40 |
| Held but < 25% of target size | 30 |
| 25–50% of target size | 20 |
| 50–75% of target size | 10 |
| >= 75% of target size | 0 |

### FACTOR 2 — DISTANCE FROM COST (30 points max)

**ZERO-SHARE RULE (DC-11 — ratified Mar 15, 2026):** For any Core position with 0 shares held, Factor 2 = 0 automatically. Factor 1 already awards maximum urgency (40 pts) for zero-share positions. A reference price must not be created for positions with no actual cost basis.

| % vs Cost | Points |
|-----------|--------|
| Below cost > 20% | 30 |
| Below cost 10–20% | 20 |
| Below cost 0–10% | 15 |
| Above cost 0–10% | 10 |
| Above cost 10–20% | 5 |
| Above cost > 20% | 0 |
| 0 shares held (no cost basis) | 0 |

### FACTOR 3 — THESIS MOMENTUM (20 points max)

| Signal | Points |
|--------|--------|
| Strong confirmed tailwind: ALL THREE of — earnings beat AND guidance raise AND sector acceleration | 20 |
| Moderate tailwind (1–2 of Strong conditions met, OR thesis intact + sector positive) | 10 |
| Neutral (thesis intact, no directional signal) | 5 |
| Headwind OR Yellow Watch active | 0 |

**SMA Validation for Factor 3:**
- Above SMA50 and SMA200: eligible for Strong or Moderate score
- Below SMA50 but above SMA200: eligible for Moderate (10 pts) only — not Strong (20 pts)
- Below SMA200: Factor 3 = 0 regardless of fundamental momentum

### FACTOR 4 — CONCENTRATION + TREND CHECK (10 points max)

| Condition | Points |
|-----------|--------|
| Position < 10% of liquid portfolio + uptrend confirmed (above SMA50) | 10 |
| Position < 10% of liquid portfolio + neutral trend | 5 |
| Position 10–20% of liquid portfolio | 3 |
| Position > 20% of liquid portfolio | 0 |

---

## SECTION 2 — CORE MATRIX EXECUTION RULES

**RULE 1 — HIGHEST SCORE WINS.**
Matrix Winner = highest composite score for the cycle. Winner claims full ammo allocation.

**RULE 2 — TIE-BREAKING:**
- Step 1: Lowest share count wins.
- Step 2 (tied share count): Lower cost basis wins.
- Step 3 (both zero shares, no cost basis): Apply DC-13 Tiebreaker (see Section 4).

**RULE 3 — NO RE-SCORING MID-CYCLE.**
Once a cycle's scores are set at /scorecard or /dca, they cannot be changed until the next cycle.

**RULE 4 — GATE STILL GOVERNS.**
Section 6.2 of doctrine_core applies independently of Matrix score. High Matrix score does NOT override gate failure.

**RULE 5 — ASTS REMAINDER IS HARDCODED.**
After primary DCA to Matrix Winner, ALL remaining ammo goes to ASTS fill target (subject to Zero-Remainder Rule — minimum 0.1 shares per order).

---

## SECTION 3 — SATELLITE PRIORITY SCORING (DC-13)

**SCOPE:** Watchtower Satellites only. Does NOT apply to Core Matrix.

**PURPOSE:** Determines deployment priority order among competing Satellite candidates.

### FACTOR S1 — EXECUTION URGENCY (10 points max)

| Condition | Points |
|-----------|--------|
| Trigger already hit + expiry window ≤14 days | 10 |
| Trigger already hit + expiry >14 days OR no expiry | 7 |
| Blackout opens ≤21 days (not yet triggered) | 5 |
| Blackout opens 21–45 days (not yet triggered) | 3 |
| No near-term time pressure | 0 |

### FACTOR S2 — THESIS CONVICTION DELTA (10 points max)

| Condition | Points |
|-----------|--------|
| Major structural re-rating event since Watchtower entry | 10 |
| Thesis intact + incremental positive catalysts since entry | 6 |
| Thesis intact, no material change since entry | 3 |
| Active ⚠️ WATCH flag (Israel, China, operational disruption) | 1 |

### FACTOR S3 — CONCENTRATION HEADROOM (10 points max)

| Condition | Points |
|-----------|--------|
| 0 shares held — full 15% headroom available | 10 |
| Partial position held — >50% of 15% cap remaining | 6 |
| Partial position held — <50% of 15% cap remaining | 3 |
| At or near 15% cap | 0 |

**Satellite scoring rules:**
- Run at /dca time alongside Core Matrix.
- Highest Satellite Priority Score = funded first from available Satellite capital.
- Ties broken by S1 (urgency) first, then Commander discretionary.
- Score logged in portfolio_state Section 4 Dashboard.

---

## SECTION 4 — DC-13 TIEBREAKER (ZERO-SHARE TIE)

Applies when two or more Core positions tie on composite score AND both hold zero shares AND fresh capital is being split.

**Step 1 — BLACKOUT PROXIMITY:**
The position with the earlier blackout window receives a minimum 55% of the fresh capital allocation. The later-blackout (or no-blackout) position receives the remainder.

**Step 2 — ADV GATE THRESHOLD:**
If blackouts are within 7 days of each other OR neither has a blackout: the position with the higher ADV gate threshold receives the larger allocation.

**Step 3 — COMMANDER DISCRETIONARY:**
If Steps 1 and 2 do not resolve the tie: Commander decides. Decision must be explicitly logged in Consensus Log before execution.

This tiebreaker applies to fresh capital splits only.

---

## SECTION 5 — DC-40 CONVICTION TIERS

**SCOPE:** Watchtower classification. Determines sub-tier status and priority.

| Tier | Description | DCA Priority |
|------|-------------|-------------|
| S1 (Core Tier 1) | Highest conviction — trigger imminent OR recurring DCA active | Competes in Core Matrix immediately |
| S2 (Core Tier 2) | High conviction — trigger not yet hit, waiting for gate clear | Competes in Core Matrix when /thesis CLEAR |
| S3 (Core Tier 3) | Monitoring — thesis confirmed, queued for next scorecard | Competes in Core Matrix at next scorecard |
| SAT-1 | Highest Satellite conviction — trigger imminent | Satellite Priority Score run immediately |
| SAT-2 | Satellite — waiting for trigger | Satellite Priority Score at /dca |
| SAT-3 | Satellite — monitoring | Satellite Priority Score at next scorecard |

**DC-13 Satellite minimum score:** SAT-1/2/3 require DC-13 Satellite Priority Score ≥ 25 to qualify for deployment in any given cycle. Score < 25 = deferred to next cycle.

---

## SECTION 6 — DC-43 REGIME SCORING MODIFIERS

**Purpose:** Adjusts Matrix scoring in extreme VIX regimes.

| Regime | VIX Range | Factor 3 Modifier | Factor 4 Modifier |
|--------|-----------|-------------------|-------------------|
| GREEN | < 22 | No change | No change |
| YELLOW | 22–24 | No change | No change |
| ORANGE | 25–30 | FREEZE — all Factor 3 = 0 | FREEZE — no new scoring |
| RED | > 30 | HARD FREEZE — Matrix suspended | HARD FREEZE |

**ORANGE/RED Freeze:** Matrix is suspended. No new DCA. Scores logged but not acted upon. When regime returns to YELLOW or GREEN: resume Matrix from last valid scores, rescore if >30 days elapsed.

**Bear Restructure (RED > 10 sessions):** All Factor 3 scores zeroed. Satellite Priority Scores suspended. Dead Hand clock frozen. Full Council review required.

---

## SECTION 7 — WATCHTOWER SCORECARD QUARTERLY PROTOCOL

Run at each quarterly War Council session (/scorecard). Full Triad required.

**Step 1:** Pull current Watchtower roster from portfolio_state.
**Step 2:** For each ticker, run full /thesis check. Update thesis dates.
**Step 3:** Score each ticker's F1-F4 (Core) or S1-S3 (Satellite).
**Step 4:** Rank Core candidates by composite score. Declare Matrix Winner for next cycle.
**Step 5:** Rank Satellite candidates by Satellite Priority Score.
**Step 6:** Review upcoming removal dates. Remove expired tickers.
**Step 7:** Evaluate admission candidates (if any). Apply full Alpha Filter + /thesis + Council vote.
**Step 8:** Log all decisions in Consensus Log. Update portfolio_state Section 2.

**Quarterly cadence:** Q1 (Jan), Q2 (Apr), Q3 (Jul), Q4 (Oct). Emergency /scorecard may be called if market regime change is sustained > 10 sessions.

---

*END OF DOCTRINE CORE MATRIX v1.0.0-BETA.1*
