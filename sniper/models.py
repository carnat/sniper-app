"""Shared data models for the Sniper OS portfolio system.

These dataclasses define the canonical schema used by both the Streamlit app
and the price pipeline, eliminating silent drift between the two systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PriceSnapshot:
    """Per-ticker enriched price data from prices.json."""
    symbol: str
    price: float = 0.0
    change: float = 0.0
    change5d: float = 0.0
    adv: float = 0.0
    volume: float = 0.0
    high52w: float = 0.0
    low52w: float = 0.0
    sma50: float = 0.0
    sma200: float = 0.0
    ytd_pct: float = 0.0
    news: list[dict] = field(default_factory=list)


@dataclass
class Position:
    """A single portfolio position."""
    symbol: str
    shares: float
    avg_cost: float
    asset_type: str = "US Stock"
    currency: str = "USD"
    master: Optional[str] = None


@dataclass
class Transaction:
    """A recorded portfolio transaction."""
    date: str
    ticker: str
    type: str  # Buy, Sell, Dividend, Fee, Split
    shares: float
    price: float
    total: float
    asset_type: str = "US Stock"
    currency: str = "USD"
    realized_pl: float = 0.0
    import_key: Optional[str] = None
    notes: Optional[str] = None
    lot_method: Optional[str] = None
    master: Optional[str] = None
    # Journal metadata
    strategy: Optional[str] = None
    session: Optional[str] = None
    direction: Optional[str] = None
    followed_rules: Optional[str] = None
    confidence: Optional[str] = None
    risk_amount: Optional[float] = None
    entry_window: Optional[str] = None
    mental_state: Optional[str] = None
    journal_note: Optional[str] = None
    r_multiple: Optional[float] = None
    # Correction tracking
    correction_of: Optional[int] = None
    reversed_by: Optional[int] = None
    correction_reason: Optional[str] = None


@dataclass
class GateStatus:
    """Market regime gate status."""
    vix: float = 0.0
    regime: str = "GREEN"
    vix_freeze: bool = False
    thb_zone: str = "B"
    fx_rate: float = 34.0
    spy_above_200dma: bool = True


@dataclass
class FundPosition:
    """A mutual fund position in the vault portfolio."""
    code: str
    units: float
    cost: float
    master: str = "N/A"
