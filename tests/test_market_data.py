"""Tests for sniper/market.py — pure functions (no API calls)."""

import unittest

import pandas as pd

from sniper.market import get_base_symbol_universe, get_stock_data


class TestGetBaseSymbolUniverse(unittest.TestCase):
    def test_combines_us_and_thai(self):
        us = {"Ticker": ["AAPL", "MSFT"]}
        thai = {"Ticker": ["ADVANC.BK"]}
        vault = []
        result = get_base_symbol_universe(us, thai, vault)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)
        self.assertIn("ADVANC.BK", result)

    def test_includes_vault_masters(self):
        us = {"Ticker": []}
        thai = {"Ticker": []}
        vault = [{"Master": "QQQ"}, {"Master": "VOO"}, {"Master": "N/A"}]
        result = get_base_symbol_universe(us, thai, vault)
        self.assertIn("QQQ", result)
        self.assertIn("VOO", result)
        self.assertNotIn("N/A", result)

    def test_includes_watchlists(self):
        us = {"Ticker": []}
        thai = {"Ticker": []}
        vault = []
        watchlists = {"Arsenal": ["VRT", "ASTS"], "WT": ["PLTR"]}
        result = get_base_symbol_universe(us, thai, vault, watchlists=watchlists)
        self.assertIn("VRT", result)
        self.assertIn("ASTS", result)
        self.assertIn("PLTR", result)

    def test_deduplicates(self):
        us = {"Ticker": ["AAPL"]}
        thai = {"Ticker": []}
        vault = []
        watchlists = {"List": ["aapl"]}
        result = get_base_symbol_universe(us, thai, vault, watchlists=watchlists)
        self.assertEqual(result.count("AAPL"), 1)

    def test_sorted_output(self):
        us = {"Ticker": ["MSFT", "AAPL", "VRT"]}
        result = get_base_symbol_universe(us, {"Ticker": []}, [])
        self.assertEqual(result, sorted(result))

    def test_strips_whitespace(self):
        us = {"Ticker": [" AAPL ", "  MSFT"]}
        result = get_base_symbol_universe(us, {"Ticker": []}, [])
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)

    def test_empty_inputs(self):
        result = get_base_symbol_universe({"Ticker": []}, {"Ticker": []}, [])
        self.assertEqual(result, [])

    def test_filters_empty_strings(self):
        us = {"Ticker": ["AAPL", "", " "]}
        result = get_base_symbol_universe(us, {"Ticker": []}, [])
        self.assertEqual(result, ["AAPL"])


class TestGetStockData(unittest.TestCase):
    def test_with_price_map(self):
        portfolio = {"Ticker": ["AAPL", "MSFT"], "Shares": [10, 5], "Avg_Cost": [150.0, 400.0]}
        price_map = {"AAPL": 160.0, "MSFT": 420.0}
        df = get_stock_data(portfolio, price_map=price_map)
        self.assertEqual(len(df), 2)
        self.assertIn("Live Price", df.columns)
        self.assertIn("P/L", df.columns)
        self.assertIn("P/L %", df.columns)
        # AAPL: value=1600, cost=1500, PL=100, PL%=6.67
        aapl_row = df[df["Ticker"] == "AAPL"].iloc[0]
        self.assertAlmostEqual(aapl_row["Live Price"], 160.0)
        self.assertAlmostEqual(aapl_row["Value"], 1600.0)
        self.assertAlmostEqual(aapl_row["Cost Basis"], 1500.0)
        self.assertAlmostEqual(aapl_row["P/L"], 100.0)

    def test_empty_portfolio(self):
        portfolio = {"Ticker": [], "Shares": [], "Avg_Cost": []}
        df = get_stock_data(portfolio, price_map={})
        self.assertEqual(len(df), 0)
        self.assertIn("Ticker", df.columns)

    def test_missing_price_defaults_to_zero(self):
        portfolio = {"Ticker": ["AAPL"], "Shares": [10], "Avg_Cost": [150.0]}
        price_map = {}  # No price for AAPL
        df = get_stock_data(portfolio, price_map=price_map)
        self.assertEqual(df.iloc[0]["Live Price"], 0.0)

    def test_zero_cost_basis_no_division_error(self):
        portfolio = {"Ticker": ["AAPL"], "Shares": [10], "Avg_Cost": [0.0]}
        price_map = {"AAPL": 150.0}
        df = get_stock_data(portfolio, price_map=price_map)
        self.assertEqual(df.iloc[0]["P/L %"], 0.0)  # Should not error


if __name__ == "__main__":
    unittest.main()
