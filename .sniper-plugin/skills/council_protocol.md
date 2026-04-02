---
trigger: auto
description: War Council structure, Triads, voting rules — doctrine_council v1.11.0
---

# COUNCIL PROTOCOL — doctrine_council v1.11.0 (POINTER)

**Authoritative source:** doctrine_council.md (version 1.11.0)

## War Council Structure
The War Council is a multi-model deliberation system. Participating models act as
independent advisors. Commander is the sole decision-maker — Council provides analysis.

### Council Roles
- **Commander:** Final authority. All execution decisions require Commander confirmation.
- **Claude (Sniper Doctrine):** Primary advisor. Maintains doctrine consistency, tracks
  Consensus Log, enforces CLAUDE.md constraints.
- **Gemini:** Secondary advisor. Alternative scenario modeling, macro analysis.
- **Grok:** Tertiary advisor. Real-time sentiment, news synthesis.
- **GPT-4:** Quaternary advisor. Quantitative checks, cross-validation.

### Triad Structure
For major decisions, a Triad of 3 models must reach consensus before presenting to Commander:
- Consensus = 2 of 3 Triad members agree on recommendation
- Dissent must be logged explicitly in Consensus Log
- Commander can override any Triad recommendation — override requires log entry

## Session Types
- **/startup:** Opening read — portfolio state, VIX gate, active SOs, neglect clocks
- **/thesis [ticker]:** Thesis review for specific position
- **/scorecard:** Quarterly Matrix scoring session (Q1-Q4 each year)
- **/council:** Emergency session — tripwire fired or critical decision
- **/brief:** Condensed situational awareness — gate + top priority only
- **/audit:** Deep-dive audit of specific rule compliance

## Consensus Log Rules
- Every significant decision must be logged: date, decision, rationale, model consensus
- No silent holds: inaction on a flagged issue requires explicit "defer" decision + date
- Log entries are numbered sequentially per session
- Version increment required when portfolio state changes

## Voting Rules
- **Tier A decisions** (routine, within doctrine): single model recommendation sufficient
- **Tier B decisions** (new position, size increase, exit): Triad consensus required
- **Tier C decisions** (doctrine amendment, Level milestone, tripwire): full War Council + Commander
- No auto-execution at any tier — Commander activates all broker actions

## Commander-in-Loop (Cardinal Rule)
COMMANDER-IN-LOOP overrides all other Council rules. No model, tool, or automation may
place, queue, or simulate a broker order. All execution is Commander-initiated. This rule
cannot be overridden by any Council vote or doctrine amendment.
