# .sniper-plugin/ — CLAUDE.md

Skills directory for the Sniper Doctrine rules-based investment system.
These files are **reference summaries only** — pointer files that summarize doctrine rules
and reference authoritative sources. They do NOT contain full doctrine text.

## Purpose
Each skill file provides a compressed, actionable reference for a specific doctrine domain.
Claude uses these files as context when assisting with investment decisions.

## Skills Index
| File | Doctrine Source | Domain |
|------|----------------|--------|
| `alpha_filters.md` | doctrine_core v1.12.1 §5 | Rules 1–4: admission gates, China watch |
| `council_protocol.md` | doctrine_council v1.11.0 | War Council, Triads, voting rules |
| `deployment_rules.md` | doctrine_core v1.12.1 §6 | Q1–Q4 gate, DC-15/16/31/32 |
| `doctrine-core.md` | doctrine_core v1.12.1 | Full core doctrine summary |
| `goal_architecture.md` | doctrine_core v1.12.1 §7 | DC-41 milestone ladder M-1 through M-4 |
| `hwm_drawdown.md` | doctrine_core v1.12.1 §6.5 | HWM tracking, Freeze, DC-59 Amber Zone |
| `matrix_scoring.md` | doctrine_core_matrix v1.0.0-BETA.1 | F1–F5 factor scoring |
| `portfolio-state.md` | portfolio_state v1.3.14 | Current positions, orders, matrix scores |
| `tripwires.md` | doctrine_core_tripwires v1.0.0-BETA.1 | Thesis break definitions |
| `vault-rules.md` | doctrine_pe v1.6.0 | Vault registry, fund positions |
| `vault_protocol.md` | doctrine_pe v1.6.0 | Rebalancing, switch triggers, E-Class |
| `victory_protocols.md` | doctrine_core v1.12.1 §6.6 | DC-49/50a/b/56/58 |
| `watchtower.md` | doctrine_ops v1.15.1 §0.4 | DC-40 tiers, roster, admission/removal |

## Constraints
- **NEVER** duplicate full doctrine text here — use POINTER references only
- **NEVER** include execution logic, broker instructions, or trade orders
- **NEVER** include secrets, API keys, or personal financial data
- Skill files should be summaries + references, not primary sources
- Per root CLAUDE.md: "Doctrine rules: authoritative full text in `docs/doctrine/`, compressed summaries in `.sniper-plugin/skills/`"

POINTER: Full doctrine text → `docs/doctrine/` (committed, version-controlled)

## Current Doctrine Versions
- project_instructions v1.3.24
- portfolio_state v1.3.14
- doctrine_ops v1.15.1
- doctrine_council v1.11.0
- doctrine_core v1.12.1
- doctrine_pe v1.6.0
- doctrine_core_matrix v1.0.0-BETA.1
- doctrine_core_tripwires v1.0.0-BETA.1
