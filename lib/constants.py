"""
SNIPER OS — Doctrine Constants
Version: v35.14 | Updated: 2026-02-24

Single source of truth for all tickers, positions, thresholds, and thesis breaks.
Import from here in gates.py, sentinel.py, and any page that needs doctrine data.
"""

# ── TICKER UNIVERSES ─────────────────────────────────────────────────────────

US_TICKERS = ["FN", "VRT", "ASTS", "VST", "MU", "APH", "ANET", "TSM", "ONDS"]
THAI_TICKERS = ["TISCO.BK", "ADVANC.BK"]
VAULT_MASTERS = ["QQQ", "VOO", "VTI", "SOXX", "GLD", "ICLN"]

# All tickers that the Sentinel scans
CORE_TICKERS = US_TICKERS + THAI_TICKERS

# ── POSITIONS (doctrine reference — not live portfolio state) ─────────────────

POSITIONS = {
    "FN":     {"shares": 0,      "avg_cost": 0,      "ref_price": 586.90},
    "VRT":    {"shares": 0.1483, "avg_cost": 235.92},
    "ASTS":   {"shares": 9.3,    "avg_cost": 74.58},
    "VST":    {"shares": 12,     "avg_cost": 158.32},
    "MU":     {"shares": 3.06,   "avg_cost": 294.01},
    "APH":    {"shares": 8,      "avg_cost": 137.27},
    "ANET":   {"shares": 3,      "avg_cost": 122.12},
    "TSM":    {"shares": 0.08,   "avg_cost": 296.42},
    "ONDS":   {"shares": 1,      "avg_cost": 9.05},
    "TISCO":  {"shares": 173,    "avg_cost": 98.61,  "currency": "THB"},
    "ADVANC": {"shares": 26,     "avg_cost": 287.13, "currency": "THB"},
}

# ── VIX GATE THRESHOLDS ───────────────────────────────────────────────────────

VIX_FREEZE   = 25   # VIX ≥ this → FREEZE  (no new deployments)
VIX_CAUTION  = 20   # VIX ≥ this → CAUTION (reduced size)
# VIX < VIX_CAUTION  → GREEN   (normal operations)

# ── FX ZONE THRESHOLDS (THB per 1 USD) ───────────────────────────────────────

FX_ZONE_A = 32   # USD/THB ≤ this → Zone A (cheap USD — dinner signal 🍽️)
FX_ZONE_C = 36   # USD/THB ≥ this → Zone C (expensive USD — hold)
# FX_ZONE_A < rate < FX_ZONE_C → Zone B (neutral)

FX_FALLBACK = 34.0  # used when live fetch fails

# ── LEVEL 1 AUDIT TARGETS ────────────────────────────────────────────────────

LEVEL_1_TARGET      = 1_000_000  # THB — true liquid target
EXTENSION_THRESHOLD = 0.10       # 10% drawdown triggers review

# ── GATE LOGIC IDENTIFIERS ───────────────────────────────────────────────────

GATE_SIGNALS = {
    "GO":      "GO",       # all gates pass
    "CAUTION": "CAUTION",  # one or more caution conditions
    "FREEZE":  "FREEZE",   # hard stop — no new deployments
}

# ── THESIS BREAK CONDITIONS (per ticker) ─────────────────────────────────────
# Reference text shown in Sentinel UI. Not enforced programmatically yet.

THESIS_BREAKS = {
    "FN": [
        "Revenue growth falls below 15% YoY",
        "Net margin compression > 5pp from peak",
        "Loss of top-5 mortgage servicer client",
    ],
    "VRT": [
        "Data-centre cooling TAM narrative reverses",
        "Gross margin < 30%",
        "Major hyperscaler cancels liquid-cooling contract",
    ],
    "ASTS": [
        "FCC licence revoked or delayed > 12 months",
        "SpaceMobile commercial launch fails",
        "Cash runway < 6 months without new financing",
    ],
    "VST": [
        "Power-purchase agreement repriced below breakeven",
        "Regulatory block on capacity expansion",
        "Data-centre occupancy < 80%",
    ],
    "MU": [
        "HBM3e design win lost to SK Hynix",
        "DRAM ASP YoY decline > 20%",
        "China export restrictions expand to HBM",
    ],
    "APH": [
        "EV/auto connector revenue declines > 10% QoQ",
        "Margin compression from commodity cost spike",
    ],
    "ANET": [
        "Hyperscaler capex guidance cut > 15%",
        "Cisco or Juniper wins back flagship cloud account",
    ],
    "TSM": [
        "Advanced node yield < 50% at 2nm ramp",
        "US/Taiwan geopolitical escalation triggers sanctions",
        "TSMC Arizona timeline slips > 18 months",
    ],
    "ONDS": [
        "Ondas drone revenue misses consensus > 30%",
        "DoD contract cancelled or not renewed",
        "Cash < $20M without clear path to profitability",
    ],
    "TISCO.BK": [
        "NPL ratio > 4%",
        "BOT policy rate cut narrows NIM > 50bps",
    ],
    "ADVANC.BK": [
        "Subscriber churn > 2% per quarter",
        "ARPU decline > 8% YoY",
        "Spectrum renewal blocked",
    ],
}
