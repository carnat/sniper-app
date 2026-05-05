"""Tests for sniper/tax_lots.py — FIFO/LIFO/Average cost lot tracking."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sniper.tax_lots import (
    get_lot_db_path,
    init_lot_database,
    lot_record_buy,
    lot_record_sell_fifo,
    lot_record_sell_lifo,
    lot_record_sell_average,
    get_lot_method_for_asset,
    lot_apply_split,
    seed_opening_lots_from_portfolios,
)


class TaxLotTestBase(unittest.TestCase):
    """Base class that patches lot DB to use a temp directory."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test_lots.db"
        self.patcher = patch("sniper.tax_lots.get_lot_db_path", return_value=self.db_path)
        self.patcher.start()
        init_lot_database()

    def tearDown(self):
        self.patcher.stop()
        self.db_path.unlink(missing_ok=True)

    def _get_lots(self, symbol=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if symbol:
            cursor.execute("SELECT * FROM tax_lots WHERE symbol = ?", (symbol,))
        else:
            cursor.execute("SELECT * FROM tax_lots")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def _get_realized(self, symbol=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if symbol:
            cursor.execute("SELECT * FROM realized_lots WHERE symbol = ?", (symbol,))
        else:
            cursor.execute("SELECT * FROM realized_lots")
        rows = cursor.fetchall()
        conn.close()
        return rows


class TestInitLotDatabase(TaxLotTestBase):
    def test_creates_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        self.assertIn("tax_lots", tables)
        self.assertIn("realized_lots", tables)
        self.assertIn("lot_metadata", tables)


class TestLotRecordBuy(TaxLotTestBase):
    def test_basic_buy(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10.0, 150.0, acquired_date="2026-01-15")
        lots = self._get_lots("AAPL")
        self.assertEqual(len(lots), 1)
        # lot_id, symbol, asset_type, currency, acquired_date, qty_orig, qty_remaining, cost, source
        self.assertEqual(lots[0][1], "AAPL")
        self.assertEqual(lots[0][5], 10.0)  # quantity_original
        self.assertEqual(lots[0][6], 10.0)  # quantity_remaining
        self.assertEqual(lots[0][7], 150.0)  # cost_per_unit

    def test_multiple_buys(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 150.0, acquired_date="2026-01-01")
        lot_record_buy("AAPL", "US Stock", "USD", 5, 160.0, acquired_date="2026-01-10")
        lots = self._get_lots("AAPL")
        self.assertEqual(len(lots), 2)

    def test_zero_quantity_ignored(self):
        lot_record_buy("AAPL", "US Stock", "USD", 0, 150.0)
        lots = self._get_lots("AAPL")
        self.assertEqual(len(lots), 0)

    def test_negative_quantity_ignored(self):
        lot_record_buy("AAPL", "US Stock", "USD", -5, 150.0)
        lots = self._get_lots("AAPL")
        self.assertEqual(len(lots), 0)


class TestFIFO(TaxLotTestBase):
    def test_fifo_basic(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 100.0, acquired_date="2026-01-01")
        lot_record_buy("AAPL", "US Stock", "USD", 10, 200.0, acquired_date="2026-01-10")
        pl = lot_record_sell_fifo("AAPL", "US Stock", "USD", 5, 150.0, sale_date="2026-02-01")
        # FIFO: sells from first lot (cost=100), profit = (150-100)*5 = 250
        self.assertAlmostEqual(pl, 250.0)

    def test_fifo_spans_multiple_lots(self):
        lot_record_buy("AAPL", "US Stock", "USD", 5, 100.0, acquired_date="2026-01-01")
        lot_record_buy("AAPL", "US Stock", "USD", 5, 200.0, acquired_date="2026-01-10")
        pl = lot_record_sell_fifo("AAPL", "US Stock", "USD", 8, 150.0, sale_date="2026-02-01")
        # First lot: 5 @ 100 → (150-100)*5 = 250
        # Second lot: 3 @ 200 → (150-200)*3 = -150
        # Total = 100
        self.assertAlmostEqual(pl, 100.0)

    def test_fifo_insufficient_lots(self):
        lot_record_buy("AAPL", "US Stock", "USD", 5, 100.0)
        pl = lot_record_sell_fifo("AAPL", "US Stock", "USD", 10, 150.0)
        self.assertIsNone(pl)

    def test_fifo_zero_quantity_sell(self):
        pl = lot_record_sell_fifo("AAPL", "US Stock", "USD", 0, 150.0)
        self.assertEqual(pl, 0.0)

    def test_fifo_creates_realized_records(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 100.0, acquired_date="2026-01-01")
        lot_record_sell_fifo("AAPL", "US Stock", "USD", 5, 120.0, sale_date="2026-02-01")
        realized = self._get_realized("AAPL")
        self.assertEqual(len(realized), 1)

    def test_fifo_updates_remaining_quantity(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 100.0, acquired_date="2026-01-01")
        lot_record_sell_fifo("AAPL", "US Stock", "USD", 3, 120.0, sale_date="2026-02-01")
        lots = self._get_lots("AAPL")
        self.assertEqual(len(lots), 1)
        self.assertAlmostEqual(lots[0][6], 7.0)  # quantity_remaining


class TestLIFO(TaxLotTestBase):
    def test_lifo_basic(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 100.0, acquired_date="2026-01-01")
        lot_record_buy("AAPL", "US Stock", "USD", 10, 200.0, acquired_date="2026-01-10")
        pl = lot_record_sell_lifo("AAPL", "US Stock", "USD", 5, 150.0, sale_date="2026-02-01")
        # LIFO: sells from second lot (cost=200), profit = (150-200)*5 = -250
        self.assertAlmostEqual(pl, -250.0)

    def test_lifo_spans_multiple_lots(self):
        lot_record_buy("AAPL", "US Stock", "USD", 5, 100.0, acquired_date="2026-01-01")
        lot_record_buy("AAPL", "US Stock", "USD", 5, 200.0, acquired_date="2026-01-10")
        pl = lot_record_sell_lifo("AAPL", "US Stock", "USD", 8, 150.0, sale_date="2026-02-01")
        # Second lot: 5 @ 200 → (150-200)*5 = -250
        # First lot: 3 @ 100 → (150-100)*3 = 150
        # Total = -100
        self.assertAlmostEqual(pl, -100.0)

    def test_lifo_insufficient_lots(self):
        lot_record_buy("AAPL", "US Stock", "USD", 5, 100.0)
        pl = lot_record_sell_lifo("AAPL", "US Stock", "USD", 10, 150.0)
        self.assertIsNone(pl)


class TestAverageCost(TaxLotTestBase):
    def test_average_basic(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 100.0, acquired_date="2026-01-01")
        lot_record_buy("AAPL", "US Stock", "USD", 10, 200.0, acquired_date="2026-01-10")
        pl = lot_record_sell_average("AAPL", "US Stock", "USD", 5, 160.0, sale_date="2026-02-01")
        # Average cost = (10*100 + 10*200) / 20 = 150
        # P/L = (160 - 150) * 5 = 50
        self.assertAlmostEqual(pl, 50.0)

    def test_average_insufficient_lots(self):
        lot_record_buy("AAPL", "US Stock", "USD", 5, 100.0)
        pl = lot_record_sell_average("AAPL", "US Stock", "USD", 10, 150.0)
        self.assertIsNone(pl)

    def test_average_zero_quantity(self):
        pl = lot_record_sell_average("AAPL", "US Stock", "USD", 0, 150.0)
        self.assertEqual(pl, 0.0)


class TestGetLotMethodForAsset(unittest.TestCase):
    def test_default_us_stock(self):
        self.assertEqual(get_lot_method_for_asset("US Stock"), "FIFO")

    def test_default_thai_stock(self):
        self.assertEqual(get_lot_method_for_asset("Thai Stock"), "FIFO")

    def test_default_mutual_fund(self):
        self.assertEqual(get_lot_method_for_asset("Mutual Fund"), "AVERAGE")

    def test_custom_policy(self):
        policies = {"US Stock": "LIFO"}
        self.assertEqual(get_lot_method_for_asset("US Stock", policies), "LIFO")

    def test_unknown_asset_type(self):
        self.assertEqual(get_lot_method_for_asset("Crypto"), "FIFO")


class TestLotApplySplit(TaxLotTestBase):
    def test_split_doubles_shares_halves_cost(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 200.0, acquired_date="2026-01-01")
        result = lot_apply_split("AAPL", "US Stock", "USD", 2.0)
        self.assertTrue(result)
        lots = self._get_lots("AAPL")
        self.assertEqual(len(lots), 1)
        self.assertAlmostEqual(lots[0][5], 20.0)  # quantity_original doubled
        self.assertAlmostEqual(lots[0][6], 20.0)  # quantity_remaining doubled
        self.assertAlmostEqual(lots[0][7], 100.0)  # cost halved

    def test_split_zero_ratio_fails(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 200.0)
        result = lot_apply_split("AAPL", "US Stock", "USD", 0)
        self.assertFalse(result)

    def test_split_negative_ratio_fails(self):
        result = lot_apply_split("AAPL", "US Stock", "USD", -1)
        self.assertFalse(result)


class TestGetLotDbPath(unittest.TestCase):
    def test_returns_path(self):
        result = get_lot_db_path()
        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path(".streamlit") / "portfolio_lots.db")

    def test_path_suffix(self):
        self.assertEqual(get_lot_db_path().suffix, ".db")


class TestSeedOpeningLots(TaxLotTestBase):
    def test_seeds_us_stocks(self):
        us = {'Ticker': ['AAPL', 'MSFT'], 'Shares': [10, 20], 'Avg_Cost': [150.0, 300.0]}
        seed_opening_lots_from_portfolios(us, {'Ticker': [], 'Shares': [], 'Avg_Cost': []}, [])
        lots = self._get_lots()
        self.assertEqual(len(lots), 2)
        symbols = {lot[1] for lot in lots}
        self.assertEqual(symbols, {'AAPL', 'MSFT'})

    def test_seeds_thai_stocks(self):
        thai = {'Ticker': ['DELTA'], 'Shares': [100], 'Avg_Cost': [50.0]}
        seed_opening_lots_from_portfolios(
            {'Ticker': [], 'Shares': [], 'Avg_Cost': []}, thai, [])
        lots = self._get_lots()
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0][1], 'DELTA')
        self.assertEqual(lots[0][2], 'Thai Stock')
        self.assertEqual(lots[0][3], 'THB')

    def test_seeds_vault_funds(self):
        vault = [{'Code': 'KFAFIX', 'Units': 500, 'Cost': 12.0}]
        seed_opening_lots_from_portfolios(
            {'Ticker': [], 'Shares': [], 'Avg_Cost': []},
            {'Ticker': [], 'Shares': [], 'Avg_Cost': []},
            vault)
        lots = self._get_lots()
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0][1], 'KFAFIX')
        self.assertEqual(lots[0][2], 'Mutual Fund')

    def test_skips_when_db_has_lots(self):
        lot_record_buy("VRT", "US Stock", "USD", 5, 100.0)
        us = {'Ticker': ['AAPL'], 'Shares': [10], 'Avg_Cost': [150.0]}
        seed_opening_lots_from_portfolios(us, {'Ticker': [], 'Shares': [], 'Avg_Cost': []}, [])
        lots = self._get_lots()
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0][1], 'VRT')

    def test_handles_empty_portfolios(self):
        seed_opening_lots_from_portfolios(
            {'Ticker': [], 'Shares': [], 'Avg_Cost': []},
            {'Ticker': [], 'Shares': [], 'Avg_Cost': []},
            [])
        lots = self._get_lots()
        self.assertEqual(len(lots), 0)

    def test_source_is_opening(self):
        us = {'Ticker': ['AAPL'], 'Shares': [10], 'Avg_Cost': [150.0]}
        seed_opening_lots_from_portfolios(us, {'Ticker': [], 'Shares': [], 'Avg_Cost': []}, [])
        lots = self._get_lots()
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0][8], 'OPENING')

    def test_zero_share_positions_skipped(self):
        us = {'Ticker': ['AAPL', 'MSFT'], 'Shares': [0, 10], 'Avg_Cost': [150.0, 300.0]}
        seed_opening_lots_from_portfolios(us, {'Ticker': [], 'Shares': [], 'Avg_Cost': []}, [])
        lots = self._get_lots()
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0][1], 'MSFT')


if __name__ == "__main__":
    unittest.main()
