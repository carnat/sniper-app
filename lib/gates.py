"""
SNIPER OS — Gate Logic (pure functions, no Streamlit imports)

Q1 — VIX Gate:    Is macro volatility safe to deploy?
Q2 — FX Gate:     Is USD/THB in a favourable zone?
Q3 — Volume Gate: Is today's volume normal (not a panic or low-liquidity session)?
Q4 — Earnings Gate: Is the ticker outside its earnings blackout window?

All functions are pure (no side effects, no network calls).
Data is injected by the caller (streamlit_app.py or tests).

Gate signal precedence: FREEZE > CAUTION > GO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lib.constants import (
    FX_FALLBACK,
    FX_ZONE_A,
    FX_ZONE_C,
    GATE_SIGNALS,
    VIX_CAUTION,
    VIX_FREEZE,
)


# ── DATA TYPES ────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result of a single gate check."""
    gate: str          # e.g. "Q1_VIX"
    signal: str        # "GO" | "CAUTION" | "FREEZE"
    value: Optional[float]
    threshold: Optional[float]
    note: str


@dataclass
class RunResult:
    """Aggregated result of running all four gates for one ticker."""
    ticker: str
    signal: str                        # worst-case signal across all gates
    gates: list[GateResult] = field(default_factory=list)

    @property
    def is_go(self) -> bool:
        return self.signal == GATE_SIGNALS["GO"]

    @property
    def is_caution(self) -> bool:
        return self.signal == GATE_SIGNALS["CAUTION"]

    @property
    def is_freeze(self) -> bool:
        return self.signal == GATE_SIGNALS["FREEZE"]


# ── VIX CLASSIFIER ───────────────────────────────────────────────────────────

def classify_vix(vix: Optional[float]) -> str:
    """
    Classify a VIX reading into a gate signal.

    Args:
        vix: Current VIX level (None treated as CAUTION — data unavailable).

    Returns:
        "GO" | "CAUTION" | "FREEZE"
    """
    if vix is None:
        return GATE_SIGNALS["CAUTION"]
    if vix >= VIX_FREEZE:
        return GATE_SIGNALS["FREEZE"]
    if vix >= VIX_CAUTION:
        return GATE_SIGNALS["CAUTION"]
    return GATE_SIGNALS["GO"]


def vix_zone_label(vix: Optional[float]) -> str:
    """Return a human-readable zone label for the VIX level."""
    if vix is None:
        return "UNKNOWN"
    if vix >= VIX_FREEZE:
        return f"🔴 FREEZE ({vix:.1f} ≥ {VIX_FREEZE})"
    if vix >= VIX_CAUTION:
        return f"🟡 CAUTION ({vix:.1f} ≥ {VIX_CAUTION})"
    return f"🟢 GREEN ({vix:.1f} < {VIX_CAUTION})"


# ── FX CLASSIFIER ────────────────────────────────────────────────────────────

def classify_fx(usd_thb: Optional[float]) -> str:
    """
    Classify a USD/THB rate into a gate signal.

    Zone A (≤ 32): cheap USD → GO (dinner signal 🍽️ — good to deploy)
    Zone B (32–36): neutral  → GO
    Zone C (≥ 36): expensive USD → CAUTION (hold, wait for THB to strengthen)

    Args:
        usd_thb: USD/THB exchange rate (THB per 1 USD). None → CAUTION.

    Returns:
        "GO" | "CAUTION"
    """
    if usd_thb is None:
        return GATE_SIGNALS["CAUTION"]
    if usd_thb >= FX_ZONE_C:
        return GATE_SIGNALS["CAUTION"]
    return GATE_SIGNALS["GO"]


def fx_zone_label(usd_thb: Optional[float]) -> str:
    """Return zone label with dinner indicator."""
    if usd_thb is None:
        return "UNKNOWN"
    if usd_thb <= FX_ZONE_A:
        return f"🍽️ Zone A — Dinner Signal ({usd_thb:.2f} ≤ {FX_ZONE_A})"
    if usd_thb < FX_ZONE_C:
        return f"🟡 Zone B — Neutral ({usd_thb:.2f})"
    return f"🔴 Zone C — Hold ({usd_thb:.2f} ≥ {FX_ZONE_C})"


# ── VOLUME GATE (Q3) ─────────────────────────────────────────────────────────

def check_volume_gate(
    today_volume: Optional[int],
    adv_20: Optional[int],
    low_volume_ratio: float = 0.3,
    spike_ratio: float = 5.0,
) -> GateResult:
    """
    Q3 — Volume Gate: flag panic-selling (spike) or thin liquidity (low).

    Args:
        today_volume: Today's traded volume.
        adv_20:       20-day average daily volume.
        low_volume_ratio:  today/adv < this → CAUTION (thin market).
        spike_ratio:       today/adv > this → CAUTION (panic / event-driven).

    Returns:
        GateResult with signal and note.
    """
    if today_volume is None or adv_20 is None or adv_20 == 0:
        return GateResult(
            gate="Q3_VOLUME",
            signal=GATE_SIGNALS["CAUTION"],
            value=None,
            threshold=None,
            note="Volume data unavailable",
        )

    ratio = today_volume / adv_20

    if ratio > spike_ratio:
        return GateResult(
            gate="Q3_VOLUME",
            signal=GATE_SIGNALS["CAUTION"],
            value=round(ratio, 2),
            threshold=spike_ratio,
            note=f"Volume spike {ratio:.1f}× ADV — possible panic/event",
        )
    if ratio < low_volume_ratio:
        return GateResult(
            gate="Q3_VOLUME",
            signal=GATE_SIGNALS["CAUTION"],
            value=round(ratio, 2),
            threshold=low_volume_ratio,
            note=f"Thin volume {ratio:.1f}× ADV — low liquidity",
        )
    return GateResult(
        gate="Q3_VOLUME",
        signal=GATE_SIGNALS["GO"],
        value=round(ratio, 2),
        threshold=None,
        note=f"Volume normal {ratio:.1f}× ADV",
    )


# ── EARNINGS GATE (Q4) ────────────────────────────────────────────────────────

def check_earnings_gate(
    days_to_earnings: Optional[int],
    blackout_days: int = 5,
) -> GateResult:
    """
    Q4 — Earnings Blackout Gate: avoid entering positions too close to earnings.

    Args:
        days_to_earnings: Calendar days until next earnings event (None if unknown).
        blackout_days:    Number of pre-earnings days to block.

    Returns:
        GateResult with signal and note.
    """
    if days_to_earnings is None:
        return GateResult(
            gate="Q4_EARNINGS",
            signal=GATE_SIGNALS["GO"],
            value=None,
            threshold=float(blackout_days),
            note="No upcoming earnings found",
        )
    if days_to_earnings <= 0:
        return GateResult(
            gate="Q4_EARNINGS",
            signal=GATE_SIGNALS["CAUTION"],
            value=float(days_to_earnings),
            threshold=float(blackout_days),
            note=f"Earnings today or past — check calendar",
        )
    if days_to_earnings <= blackout_days:
        return GateResult(
            gate="Q4_EARNINGS",
            signal=GATE_SIGNALS["CAUTION"],
            value=float(days_to_earnings),
            threshold=float(blackout_days),
            note=f"Earnings in {days_to_earnings}d — inside {blackout_days}d blackout",
        )
    return GateResult(
        gate="Q4_EARNINGS",
        signal=GATE_SIGNALS["GO"],
        value=float(days_to_earnings),
        threshold=float(blackout_days),
        note=f"Earnings in {days_to_earnings}d — outside blackout",
    )


# ── FULL GATE RUN ────────────────────────────────────────────────────────────

def _worst_signal(*signals: str) -> str:
    """Return the worst signal across all provided signals (FREEZE > CAUTION > GO)."""
    precedence = {
        GATE_SIGNALS["FREEZE"]:  2,
        GATE_SIGNALS["CAUTION"]: 1,
        GATE_SIGNALS["GO"]:      0,
    }
    return max(signals, key=lambda s: precedence.get(s, 0))


def run_gates(
    ticker: str,
    vix: Optional[float],
    usd_thb: Optional[float],
    today_volume: Optional[int] = None,
    adv_20: Optional[int] = None,
    days_to_earnings: Optional[int] = None,
    blackout_days: int = 5,
) -> RunResult:
    """
    Run all four gates for a given ticker and return an aggregated RunResult.

    Args:
        ticker:           Ticker symbol (label only — no network calls here).
        vix:              Current VIX level.
        usd_thb:          USD/THB exchange rate.
        today_volume:     Today's volume for Q3.
        adv_20:           20-day ADV for Q3.
        days_to_earnings: Days until next earnings for Q4.
        blackout_days:    Pre-earnings blackout window in days.

    Returns:
        RunResult with per-gate breakdown and overall signal.
    """
    # Q1 — VIX
    vix_signal = classify_vix(vix)
    vix_note = vix_zone_label(vix)
    q1 = GateResult(
        gate="Q1_VIX",
        signal=vix_signal,
        value=vix,
        threshold=float(VIX_CAUTION),
        note=vix_note,
    )

    # Q2 — FX
    fx_signal = classify_fx(usd_thb)
    fx_note = fx_zone_label(usd_thb)
    q2 = GateResult(
        gate="Q2_FX",
        signal=fx_signal,
        value=usd_thb,
        threshold=float(FX_ZONE_C),
        note=fx_note,
    )

    # Q3 — Volume
    q3 = check_volume_gate(today_volume, adv_20)

    # Q4 — Earnings
    q4 = check_earnings_gate(days_to_earnings, blackout_days)

    overall = _worst_signal(q1.signal, q2.signal, q3.signal, q4.signal)

    return RunResult(
        ticker=ticker,
        signal=overall,
        gates=[q1, q2, q3, q4],
    )
