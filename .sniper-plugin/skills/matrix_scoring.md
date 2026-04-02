---
trigger: auto
description: Matrix scoring — F1-F5 factors, doctrine_core_matrix v1.0.0
---

# MATRIX SCORING — doctrine_core_matrix v1.0.0 (POINTER)

**Authoritative source:** doctrine_core_matrix.md (version 1.0.0)

## Overview
The Matrix is a 5-factor scoring system (F1–F5) used to rank Arsenal candidates
and determine DCA deployment priority. Max score: 100 points.
Scored at each quarterly War Council scorecard session.

## Factors

### F1 — Business Quality (0 / 20 / 40)
Scores the quality of the underlying business:
- 40: World-class structural moat, dominant in a growing secular trend
- 20: Strong business with competitive differentiation, sector leader
- 0: Thesis break or disqualifier active (Yellow Watch, China concern, etc.)

### F2 — Financial Transparency (0 / 10 / 20)
Scores the clarity and quality of reported financials:
- 20: Clean GAAP, simple capital structure, high disclosure quality
- 10: Minor complexity (e.g., adjustments needed, one-time items)
- 0: Restated financials, GAAP/Adj divergence >30%, or F2 disqualifier

### F3 — Market Position / Momentum (0 / 10 / 15 / 20)
Scores current price vs SMA and relative sector momentum:
- 20: Above SMA50 and SMA200, outperforming sector ETF YTD
- 15: Above SMA200 only, or at SMA50 support
- 10: Below SMA50 but above SMA200 with recovering trend
- 0: Yellow Watch active OR below SMA200 with declining trend

### F4 — Deployment Readiness (0 / 5 / 10)
Scores how execution-ready the position is:
- 10: Trigger clear, ADV gate met, thesis current, no blackout
- 5: Minor constraint (approaching blackout, ADV borderline)
- 0: Hard block (VIX freeze, blackout active, thesis expired)

### F5 — Commander Conviction Bonus (0 / 5 / 10 / 20)
Discretionary conviction bonus assigned by Commander at scorecard session:
- 20: Extremely high conviction — asymmetric upside, highest portfolio priority
- 10: High conviction — standard Core candidate
- 5: Moderate conviction — Satellite or monitoring position
- 0: Low conviction / under review

## Current Matrix Scores (Mar 16, 2026 Final)
| Rank | Ticker | F1 | F2 | F3 | F4 | Total | Notes |
|------|--------|----|----|----|----|-------|-------|
| 🥇 | FN   | 40 | 0  | 20 | 10 | 70    | Tiebreak: BWXT wins (earlier blackout + higher ADV) |
| 🥇 | BWXT | 40 | 0  | 20 | 10 | 70    | Deploy ฿11,000 · Core★ |
| 3  | APH  | 20 | 20 | 15 | 10 | 65    | ⚠️ CHINA WATCH 16.3% |
| 4  | VRT/TSM | — | — | — | — | 50 | VRT Yellow Watch F3=0 · TSM Israel Watch |

## Scoring Rules
- Tiebreak: earlier blackout date wins (more urgent to deploy before window closes)
- Secondary tiebreak: higher ADV gate (more liquid = less execution risk)
- Score of 0 in any factor does NOT disqualify — position still held; DCA frozen until resolved
- Re-scored at each quarterly War Council session
