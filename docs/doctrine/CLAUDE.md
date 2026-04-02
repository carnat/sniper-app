# docs/doctrine/ — CLAUDE.md

## What This Directory Contains

These files contain **Sniper Doctrine RULES ONLY** — no sensitive portfolio data.

They are the public, version-controlled rulebook for the Sniper Doctrine rules-based investment system.
Committed to Git. Safe to read, reference, and amend via PR.

## What Is NOT Here

Sensitive portfolio state lives elsewhere (gitignored):
- Actual positions, share counts, cost bases → `data/portfolio_config.json` (gitignored)
- Generated portfolio JSON → `data/portfolio.json` (gitignored)
- Vault NAVs, ammo stack, HWM value → `data/portfolio_config.json` or `command_center/private.json`

## Version Registry

| File | Version | Source Doctrine |
|------|---------|----------------|
| `doctrine_core.md` | v1.12.1 | doctrine_core_v1_12_1 |
| `doctrine_ops.md` | v1.15.1 | doctrine_ops_v1_15_1 |
| `doctrine_council.md` | v1.11.0 | doctrine_council_v1_11_0 |
| `doctrine_pe.md` | v1.6.0 | doctrine_pe_v1_6_0 |
| `doctrine_core_tripwires.md` | v1.0.0-BETA.1 | doctrine_core_tripwires_v1_0_0_BETA_1 |
| `doctrine_core_matrix.md` | v1.0.0-BETA.1 | doctrine_core_matrix_v1_0_0_BETA_1 |

## Relationship to `.sniper-plugin/skills/`

The skills files (`.sniper-plugin/skills/`) are **compressed summaries** for Claude context injection.
The doctrine files here are the **authoritative full text** primary sources.

POINTER: Full doctrine text → `docs/doctrine/` (committed, version-controlled)
POINTER: Compressed summaries → `.sniper-plugin/skills/` (context injection only)

## Constraints

- **NEVER** store actual portfolio values, share counts, cost bases, or ammo here
- **NEVER** store fund NAV values or vault positions here
- **NEVER** store HWM actual values here
- Rules, formulas, and procedures are safe — they contain no personal financial data
- Doctrine amendments follow the 5-step PE merge procedure in `doctrine_pe.md`
