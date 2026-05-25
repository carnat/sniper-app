"""Extended tests for sniper.tax_lots."""

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sniper.tax_lots import (
    get_lot_method_for_asset,
    init_lot_database,
    lot_apply_split,
    lot_record_buy,
    lot_record_sell_fifo,
    seed_opening_lots_from_portfolios,
)


class TaxLotsExtendedTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test_lots.db"
        self.patcher = patch("sniper.tax_lots.get_lot_db_path", return_value=self.db_path)
        self.patcher.start()
        init_lot_database()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def fetch_lots(self, symbol=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if symbol:
            cursor.execute(
                "SELECT symbol, asset_type, currency, quantity_original, quantity_remaining, cost_per_unit, source FROM tax_lots WHERE symbol = ? ORDER BY rowid ASC",
                (symbol,),
            )
        else:
            cursor.execute(
                "SELECT symbol, asset_type, currency, quantity_original, quantity_remaining, cost_per_unit, source FROM tax_lots ORDER BY rowid ASC"
            )
        rows = cursor.fetchall()
        conn.close()
        return rows


class TestGetLotMethodForAssetExtended(unittest.TestCase):
    def test_defaults_and_custom_policies(self):
        self.assertEqual(get_lot_method_for_asset("US Stock"), "FIFO")
        self.assertEqual(get_lot_method_for_asset("Mutual Fund"), "AVERAGE")
        self.assertEqual(get_lot_method_for_asset("Crypto"), "FIFO")
        self.assertEqual(
            get_lot_method_for_asset("US Stock", {"US Stock": "lifo", "Mutual Fund": "fifo"}),
            "LIFO",
        )


class TestLotApplySplitExtended(TaxLotsExtendedTestBase):
    def test_split_doubles_quantity_and_halves_cost(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 200.0, acquired_date="2026-01-01")

        result = lot_apply_split("AAPL", "US Stock", "USD", 2.0)

        self.assertTrue(result)
        lot = self.fetch_lots("AAPL")[0]
        self.assertAlmostEqual(lot[3], 20.0)
        self.assertAlmostEqual(lot[4], 20.0)
        self.assertAlmostEqual(lot[5], 100.0)

    def test_split_returns_false_for_zero_or_negative_ratio(self):
        lot_record_buy("AAPL", "US Stock", "USD", 10, 200.0, acquired_date="2026-01-01")

        self.assertFalse(lot_apply_split("AAPL", "US Stock", "USD", 0))
        self.assertFalse(lot_apply_split("AAPL", "US Stock", "USD", -2))

    def test_split_only_affects_open_lots(self):
        lot_record_buy("AAPL", "US Stock", "USD", 5, 100.0, acquired_date="2026-01-01")
        lot_record_buy("AAPL", "US Stock", "USD", 5, 200.0, acquired_date="2026-01-10")
        lot_record_sell_fifo("AAPL", "US Stock", "USD", 5, 150.0, sale_date="2026-02-01")

        result = lot_apply_split("AAPL", "US Stock", "USD", 2.0)

        self.assertTrue(result)
        first_lot, second_lot = self.fetch_lots("AAPL")
        self.assertAlmostEqual(first_lot[3], 5.0)
        self.assertAlmostEqual(first_lot[4], 0.0)
        self.assertAlmostEqual(first_lot[5], 100.0)
        self.assertAlmostEqual(second_lot[3], 10.0)
        self.assertAlmostEqual(second_lot[4], 10.0)
        self.assertAlmostEqual(second_lot[5], 100.0)


class TestSeedOpeningLotsExtended(TaxLotsExtendedTestBase):
    def test_seeds_us_thai_and_vault_positions_when_database_is_empty(self):
        us_portfolio = {"Ticker": ["AAPL"], "Shares": [10], "Avg_Cost": [150.0]}
        thai_stocks = {"Ticker": ["ADVANC.BK"], "Shares": [20], "Avg_Cost": [220.0]}
        vault_portfolio = [{"Code": "SCBNDQ(E)", "Units": 1000, "Cost": 13.5}]

        seed_opening_lots_from_portfolios(us_portfolio, thai_stocks, vault_portfolio)

        lots = self.fetch_lots()
        self.assertEqual(len(lots), 3)
        self.assertEqual(
            lots,
            [
                ("AAPL", "US Stock", "USD", 10.0, 10.0, 150.0, "OPENING"),
                ("ADVANC.BK", "Thai Stock", "THB", 20.0, 20.0, 220.0, "OPENING"),
                ("SCBNDQ(E)", "Mutual Fund", "THB", 1000.0, 1000.0, 13.5, "OPENING"),
            ],
        )

    def test_seed_is_idempotent_when_database_already_has_lots(self):
        lot_record_buy("VRT", "US Stock", "USD", 5, 100.0, acquired_date="2026-01-01")

        seed_opening_lots_from_portfolios(
            {"Ticker": ["AAPL"], "Shares": [10], "Avg_Cost": [150.0]},
            {"Ticker": ["ADVANC.BK"], "Shares": [20], "Avg_Cost": [220.0]},
            [{"Code": "SCBNDQ(E)", "Units": 1000, "Cost": 13.5}],
        )

        lots = self.fetch_lots()
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0][0], "VRT")

    def test_seed_skips_zero_share_entries(self):
        seed_opening_lots_from_portfolios(
            {"Ticker": ["AAPL", "MSFT"], "Shares": [0, 10], "Avg_Cost": [150.0, 300.0]},
            {"Ticker": ["ADVANC.BK"], "Shares": [0], "Avg_Cost": [220.0]},
            [{"Code": "SCBNDQ(E)", "Units": 0, "Cost": 13.5}],
        )

        lots = self.fetch_lots()
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0][0], "MSFT")


class TestLotOperationsExtended(TaxLotsExtendedTestBase):
    def test_lot_record_buy_with_zero_quantity_does_nothing(self):
        lot_record_buy("AAPL", "US Stock", "USD", 0, 150.0)

        self.assertEqual(self.fetch_lots(), [])

    def test_lot_record_sell_fifo_with_zero_quantity_returns_zero(self):
        self.assertEqual(lot_record_sell_fifo("AAPL", "US Stock", "USD", 0, 150.0), 0.0)

    def test_sell_exact_total_available_succeeds_with_floating_point_tolerance(self):
        lot_record_buy("AAPL", "US Stock", "USD", 0.1, 10.0, acquired_date="2026-01-01")
        lot_record_buy("AAPL", "US Stock", "USD", 0.2, 20.0, acquired_date="2026-01-02")

        realized = lot_record_sell_fifo("AAPL", "US Stock", "USD", 0.3, 15.0, sale_date="2026-02-01")

        self.assertIsNotNone(realized)
        self.assertAlmostEqual(realized, -0.5)
        remaining_total = sum(lot[4] for lot in self.fetch_lots("AAPL"))
        self.assertLessEqual(abs(remaining_total), 1e-9)


if __name__ == "__main__":
    unittest.main()
