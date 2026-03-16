---
trigger: auto
description: Sniper Vault Registry — fund positions, exposure map, E-class liquid summary, overlap flags
---

# SNIPER VAULT REGISTRY v1.0.1

**LAST CONFIRMED:** Mar 9, 2026 — SCB + Krungsri screenshots verified Mar 9 session
**DOC VERSION:** v1.0.1
**LAST EDITED:** Mar 13, 2026
**STATUS:** Live. Replaces vault_registry_v1_0_0.md

---

## CHANGELOG (LAST 3 ENTRIES)

- **[Mar 13, 2026] v1.0.1:** INTEGRITY rule (cost vs price) converted to POINTER → doctrine_ops (current version) Section 0.3. /review instruction converted to POINTER → doctrine_ops (current version) Section 0.3. Watchtower overlap table updated with PLTR, TSEM, NBIS entries. Token savings ~300.
- **[Mar 9, 2026] v1.0.0:** Baseline. Thai fund tables, exposure map, overlap flags, rebalancing protocol.

---

POINTER: INTEGRITY rule (cost vs price interpretation) → doctrine_ops (current version) Section 0.3
POINTER: /review command definition and YTD sweep protocol → doctrine_ops (current version) Section 0.3
POINTER: Vault rules (locked 20 years, no cash redemption from SSF/RMF) → doctrine_core (current version) SUB-SECTION 6.3
POINTER: Vault Rebalancing Protocol (switch triggers, evaluation framework) → doctrine_core (current version) SUB-SECTION 6.8
POINTER: Vault switch settlement status → portfolio_state (current version) Section 4
POINTER: E-Class fund liquidity tiering (Tier 2 at 80%) → doctrine_core (current version) Section 7
POINTER: War Council Intel Brief protocol for vault benchmarks → doctrine_council (current version) SUB-SECTION 0.5

---

## SECTION 4.1 — THE VAULT (LOCKED FUNDS + MASTER LINKS)

### SCB ACTIVE POSITIONS

| FUND | UNITS | COST/UNIT (฿) | TOTAL COST (฿) | CURRENT VALUE (฿) | NAV DATE | RETURN | WRAPPER |
|------|-------|--------------|----------------|-------------------|----------|--------|---------|
| SCB70-SSFX | 899.7653 | ฿11.1140 | ฿10,000.00 | ฿10,158.53 | Mar 6 | +1.59% | SSF 🔒 |
| SCBKEQTGE | 748.5191 | ฿21.7677 | ฿16,293.54 | ฿13,606.43 | Mar 5 | -16.49% | E-Class 💧 |
| SCBRMDIGI(A) | 550.6086 | ฿14.5294 | ฿8,000.00 | ฿9,701.50 | Mar 4 | +21.27% | RMF 🔒 |
| SCBRMNDQ(A) | 1,204.8211 | ฿12.4500 | ฿15,000.00 | ฿17,596.17 | Mar 5 | +17.31% | RMF 🔒 |
| SCBRMS&P500 | 367.5736 | ฿16.3233 | ฿6,000.00 | ฿7,590.28 | Mar 5 | +26.50% | RMF 🔒 |
| SCBS&P500(SSFA) | 390.8886 | ฿30.6993 | ฿12,000.00 | ฿13,403.37 | Mar 5 | +11.69% | SSFA 🔒 |
| SCBS&P500(SSFE) | 768.1249 | ฿29.7400 | ฿22,844.02 | ฿24,503.88 | Mar 5 | +7.27% | SSFE 🔒 |
| SCBS&P500-SSF | 1,196.8442 | ฿23.4604 | ฿28,078.48 | ฿38,493.02 | Mar 5 | +37.09% | SSF 🔒 |
| SCBS&P500(E) | 1,292.3143 | ฿38.0000 | ฿49,107.90 | ฿51,577.04 | Mar 5 | +5.03% | E-Class 💧 |
| SCBSEMI(E) | 2,191.8427 | ฿24.1148 | ฿52,855.88 | ฿53,909.25 | Mar 5 | +1.99% | E-Class 💧 |
| SCBSEMI(SSF) | 1,912.9751 | ฿16.7279 | ฿32,000.00 | ฿38,739.47 | Mar 5 | +21.06% | SSF 🔒 |
| SCBSEMI(SSFE) | 2,384.6634 | ฿17.7828 | ฿42,405.98 | ฿50,362.42 | Mar 5 | +18.76% | SSFE 🔒 |
| SCBGOLDHE | 100.3480 | ฿14.9480 | ฿1,500.00 | ฿2,373.37 | Mar 6 | +58.22% | E-Class 💧 |

**SCB Total Cost: ฿296,085.80 | SCB Current Value: ฿332,014.73 | Return: +12.13%**

NAV Note: Values reflect Mar 5–6 closes. Mar 6 broad selloff (VIX 29.49) will pressure next NAV cycle.

**WRAPPER KEY:**
- 🔒 = Locked (SSF/SSFA/SSFE/RMF) — no cash redemption, Tier 3 for Level 1.
- 💧 = Liquid (E-Class) — switchable and redeemable, Tier 2 (80%) for Level 1.

### KRUNGSRI ACTIVE POSITIONS

| FUND | UNITS | COST/UNIT (฿) | TOTAL COST (฿) | CURRENT VALUE (฿) | NAV DATE | RETURN | WRAPPER |
|------|-------|--------------|----------------|-------------------|----------|--------|---------|
| KFGTECHRMF | 2,196.8245 | ฿18.4255 | ฿40,477.59 | ฿40,537.34 | Mar 5 | +0.15% | RMF 🔒 |
| KFUSSSF | 4,027.3086 | ฿6.1584 | ฿24,801.78 | ฿24,801.78 | Mar 4 | 0% (new) | SSF 🔒 |

**Krungsri Total Cost: ฿65,279.37 | Krungsri Current Value: ฿65,339.12 | Return: +0.09%**

### VAULT TOTALS

| | COST | CURRENT VALUE | RETURN |
|---|---|---|---|
| SCB | ฿296,085.80 | ฿332,014.73 | +12.13% |
| Krungsri | ฿65,279.37 | ฿65,339.12 | +0.09% |
| **TOTAL** | **฿361,365.17** | **฿397,353.85** | **+9.95%** |

**VAULT SWITCH STATUS: 5/5 COMPLETE ✅** All switches confirmed settled as of Mar 9, 2026 session.

---

## VAULT EXPOSURE MAP (Post-Switch, All 5 Settled)

| EXPOSURE | FUNDS | WRAPPER | LIQUID? | ROLE |
|----------|-------|---------|---------|------|
| **SOXX** (Semiconductors) | SCBSEMI(E) + SCBSEMI(SSF) + SCBSEMI(SSFE) | E+SSF+SSFE | Partial 💧 | **PRIMARY THESIS** |
| **VOO** (S&P 500) | SCBS&P500(E) + SCBS&P500-SSF + SCBS&P500(SSFA) + SCBS&P500(SSFE) + SCBRMS&P500 | E+SSF+SSFA+SSFE+RMF | Partial 💧 | **BROAD BETA** |
| **QQQ** (Nasdaq 100) | SCBRMDIGI(A) + SCBRMNDQ(A) | RMF+RMF | 🔒 None | **REDUCED** |
| **EWY** (South Korea) | SCBKEQTGE | E-Class | 💧 Yes | **MEMORY PLAY** |
| **Global Tech** | KFGTECHRMF | RMF | 🔒 None | **GROWTH TILT** |
| **US Equity** | KFUSSSF | SSF | 🔒 None | **BROAD US** |
| **SET** (Thai Market) | SCB70-SSFX | SSF | 🔒 None | **LOCAL ANCHOR** |
| **Gold** | SCBGOLDHE | E-Class | 💧 Yes | **HEDGE** |

---

## E-CLASS LIQUID SUMMARY (Tier 2 for Level 1)

| FUND | CURRENT VALUE | TIER 2 CONTRIBUTION (×0.80) |
|------|---------------|------------------------------|
| SCBKEQTGE | ฿13,606.43 | ฿10,885.14 |
| SCBS&P500(E) | ฿51,577.04 | ฿41,261.63 |
| SCBSEMI(E) | ฿53,909.25 | ฿43,127.40 |
| SCBGOLDHE | ฿2,373.37 | ฿1,898.70 |
| **E-Class Total** | **฿121,466.09** | **฿97,172.87** |

---

## WATCHTOWER VAULT OVERLAP CHECK

At each Watchtower graduation via /thesis, IO confirms vault overlap status and logs the result in portfolio_state Section 2 Watchtower entry for that ticker.

**OVERLAP CLASSIFICATIONS:**
- **ZERO** = Not present in any Vault index. No constraint.
- **MINIMAL** = Present but < 1% weight. Acceptable. Note in Watchtower entry.
- **MODERATE** = 1%–5% weight. Flag for awareness. No block.
- **SIGNIFICANT** = > 5% weight. Flag for /council rebalance review before deployment.

| TICKER | OVERLAP | NOTES |
|--------|---------|-------|
| AAOI | ZERO | Not present in any Vault index. |
| PLTR | ZERO | Not present in any Vault index. Software platform — no hardware Vault exposure. |
| TSEM | MINIMAL | Specialty mid-cap (~$13B). < 1% weight in SCBSEMI. Does not breach significant threshold. |
| NBIS | ZERO | Newly restructured entity. No legacy footprint in any Vault index. |

---

## CORE DCA OVERLAP FLAG (Post-Switch)

- 🔴 **MU:** EXTREME overlap (SOXX ×3 + EWY + VOO + QQQ remnants). LOWEST DCA priority.
- 🔴 **TSM:** HIGH overlap (SOXX ×3 + ACWI trace). Low DCA priority.
- 🟡 **ANET:** MODERATE overlap (QQQ remnants + VOO). Mid-pack.
- 🟢 **APH:** LOW overlap (VOO only, small weight). Differentiated via defense angle.
- 🟢 **VST:** MINIMAL overlap. Nuclear = unique.
- 🟢 **VRT:** MINIMAL overlap. Data center cooling = unique.
- 🟢 **FN:** ZERO overlap (not in ANY Vault fund). Most differentiated play in entire portfolio.

---

## COST BASIS vs YTD PERFORMANCE NOTE

% vs Cost reflects ENTRY TIMING, not YTD asset performance. Never confuse a cost-basis problem for a thesis problem. Run /review command to see master ETF YTD performance separately from cost basis.

---

## VAULT REBALANCING WATCH LIST

| FUND | WATCH REASON | RESOLUTION DATE | ACTION |
|------|-------------|-----------------|--------|
| SCBKEQTGE | Samsung/Nvidia HBM watch + Korea oil import exposure (Iran war). Mar 18 MU earnings = dual resolution event for HBM thesis AND switch decision. | Mar 18, 2026 | HOLD — await MU earnings |
| SCBRMDIGI + SCBRMNDQ | QQQ remnants — high SOXX overlap, long-horizon switch candidates. | No urgency | HOLD — RMF, tax implications |

---

*END OF VAULT REGISTRY v1.0.1*
