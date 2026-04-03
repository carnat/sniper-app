"""Tests for sniper/models.py — dataclass creation, defaults, and field access."""

import unittest
from dataclasses import asdict

from sniper.models import (
    PriceSnapshot,
    Position,
    Transaction,
    GateStatus,
    FundPosition,
)


class TestPriceSnapshot(unittest.TestCase):
    def test_create_with_defaults(self):
        snap = PriceSnapshot(symbol="AAPL")
        self.assertEqual(snap.symbol, "AAPL")
        self.assertEqual(snap.price, 0.0)
        self.assertEqual(snap.change, 0.0)
        self.assertEqual(snap.change5d, 0.0)
        self.assertEqual(snap.adv, 0.0)
        self.assertEqual(snap.volume, 0.0)
        self.assertEqual(snap.high52w, 0.0)
        self.assertEqual(snap.low52w, 0.0)
        self.assertEqual(snap.sma50, 0.0)
        self.assertEqual(snap.sma200, 0.0)
        self.assertEqual(snap.ytd_pct, 0.0)
        self.assertEqual(snap.news, [])

    def test_create_with_values(self):
        snap = PriceSnapshot(
            symbol="VRT", price=150.5, change=2.3, change5d=-1.1,
            adv=500000, volume=700000, high52w=180.0, low52w=90.0,
            sma50=145.0, sma200=130.0, ytd_pct=15.5,
            news=[{"title": "test"}],
        )
        self.assertEqual(snap.symbol, "VRT")
        self.assertEqual(snap.price, 150.5)
        self.assertEqual(snap.news, [{"title": "test"}])

    def test_news_list_is_independent(self):
        snap1 = PriceSnapshot(symbol="A")
        snap2 = PriceSnapshot(symbol="B")
        snap1.news.append({"title": "only for A"})
        self.assertEqual(snap2.news, [])

    def test_asdict(self):
        snap = PriceSnapshot(symbol="TSM", price=100.0)
        d = asdict(snap)
        self.assertEqual(d["symbol"], "TSM")
        self.assertEqual(d["price"], 100.0)
        self.assertIsInstance(d, dict)


class TestPosition(unittest.TestCase):
    def test_create_with_defaults(self):
        pos = Position(symbol="AAPL", shares=10.0, avg_cost=150.0)
        self.assertEqual(pos.symbol, "AAPL")
        self.assertEqual(pos.shares, 10.0)
        self.assertEqual(pos.avg_cost, 150.0)
        self.assertEqual(pos.asset_type, "US Stock")
        self.assertEqual(pos.currency, "USD")
        self.assertIsNone(pos.master)

    def test_create_thai_stock(self):
        pos = Position(symbol="ADVANC.BK", shares=100, avg_cost=220.0,
                       asset_type="Thai Stock", currency="THB")
        self.assertEqual(pos.asset_type, "Thai Stock")
        self.assertEqual(pos.currency, "THB")

    def test_create_with_master(self):
        pos = Position(symbol="SCBNDQ", shares=1000, avg_cost=13.5,
                       asset_type="Mutual Fund", currency="THB", master="QQQ")
        self.assertEqual(pos.master, "QQQ")


class TestTransaction(unittest.TestCase):
    def test_create_with_defaults(self):
        txn = Transaction(date="2026-01-15", ticker="AAPL", type="Buy",
                          shares=10.0, price=190.5, total=1905.0)
        self.assertEqual(txn.asset_type, "US Stock")
        self.assertEqual(txn.currency, "USD")
        self.assertEqual(txn.realized_pl, 0.0)
        self.assertIsNone(txn.import_key)
        self.assertIsNone(txn.notes)
        self.assertIsNone(txn.strategy)
        self.assertIsNone(txn.correction_of)

    def test_journal_metadata_fields(self):
        txn = Transaction(
            date="2026-01-15", ticker="VRT", type="Buy",
            shares=5, price=100, total=500,
            strategy="DCA", session="GREEN",
            direction="Long", confidence="High",
            risk_amount=250.0, mental_state="Focused",
        )
        self.assertEqual(txn.strategy, "DCA")
        self.assertEqual(txn.session, "GREEN")
        self.assertEqual(txn.risk_amount, 250.0)

    def test_correction_tracking(self):
        txn = Transaction(
            date="2026-02-01", ticker="AAPL", type="Sell",
            shares=2, price=201.2, total=402.4,
            correction_of=1, correction_reason="Wrong price",
        )
        self.assertEqual(txn.correction_of, 1)
        self.assertEqual(txn.correction_reason, "Wrong price")


class TestGateStatus(unittest.TestCase):
    def test_defaults(self):
        gate = GateStatus()
        self.assertEqual(gate.vix, 0.0)
        self.assertEqual(gate.regime, "GREEN")
        self.assertFalse(gate.vix_freeze)
        self.assertEqual(gate.thb_zone, "B")
        self.assertEqual(gate.fx_rate, 34.0)
        self.assertTrue(gate.spy_above_200dma)

    def test_red_regime(self):
        gate = GateStatus(vix=35.0, regime="RED", vix_freeze=True)
        self.assertEqual(gate.regime, "RED")
        self.assertTrue(gate.vix_freeze)


class TestFundPosition(unittest.TestCase):
    def test_defaults(self):
        fund = FundPosition(code="SCBNDQ(E)", units=1000, cost=13.5)
        self.assertEqual(fund.code, "SCBNDQ(E)")
        self.assertEqual(fund.units, 1000)
        self.assertEqual(fund.cost, 13.5)
        self.assertEqual(fund.master, "N/A")

    def test_with_master(self):
        fund = FundPosition(code="SCBS&P500FUND(E)", units=200, cost=38.2, master="VOO")
        self.assertEqual(fund.master, "VOO")


if __name__ == "__main__":
    unittest.main()
