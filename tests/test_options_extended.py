"""Extended tests for sniper/options.py — record_atm_iv and edge cases."""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from sniper.options import (
    normalize_market_symbol,
    record_atm_iv,
    compute_iv_rank_percentile,
    estimate_atm_iv,
    annotate_black_scholes_greeks,
)


class TestNormalizeMarketSymbol(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(normalize_market_symbol("aapl"), "AAPL")

    def test_strips_whitespace(self):
        self.assertEqual(normalize_market_symbol("  msft  "), "MSFT")

    def test_empty_string(self):
        self.assertEqual(normalize_market_symbol(""), "")

    def test_none(self):
        self.assertEqual(normalize_market_symbol(None), "")

    def test_thai_stock_suffix(self):
        self.assertEqual(normalize_market_symbol("advanc.bk"), "ADVANC.BK")


class TestRecordAtmIv(unittest.TestCase):
    def test_records_new_entry(self):
        history = {}
        result = record_atm_iv("AAPL", 0.25, history)
        self.assertIn("AAPL", result)
        self.assertEqual(len(result["AAPL"]), 1)
        self.assertAlmostEqual(result["AAPL"][0]["atm_iv"], 0.25)

    def test_replaces_same_day(self):
        today = datetime.now().strftime("%Y-%m-%d")
        history = {"AAPL": [{"date": today, "atm_iv": 0.20}]}
        result = record_atm_iv("AAPL", 0.30, history)
        self.assertEqual(len(result["AAPL"]), 1)
        self.assertAlmostEqual(result["AAPL"][0]["atm_iv"], 0.30)

    def test_none_iv_returns_unchanged(self):
        history = {"AAPL": [{"date": "2026-01-01", "atm_iv": 0.20}]}
        result = record_atm_iv("AAPL", None, history)
        self.assertEqual(len(result["AAPL"]), 1)

    def test_empty_symbol_returns_unchanged(self):
        history = {}
        result = record_atm_iv("", 0.25, history)
        self.assertEqual(result, {})

    def test_normalizes_symbol(self):
        history = {}
        result = record_atm_iv(" aapl ", 0.25, history)
        self.assertIn("AAPL", result)

    def test_caps_at_400_entries(self):
        entries = [{"date": f"2025-{i:04d}", "atm_iv": 0.2} for i in range(500)]
        history = {"AAPL": entries}
        result = record_atm_iv("AAPL", 0.30, history)
        self.assertLessEqual(len(result["AAPL"]), 400)


class TestComputeIvRankPercentileEdgeCases(unittest.TestCase):
    def test_none_current_iv(self):
        iv_rank, iv_pct, count = compute_iv_rank_percentile("AAPL", None, {})
        self.assertIsNone(iv_rank)
        self.assertIsNone(iv_pct)
        self.assertEqual(count, 0)

    def test_empty_history(self):
        iv_rank, iv_pct, count = compute_iv_rank_percentile("AAPL", 0.25, {})
        self.assertIsNone(iv_rank)
        self.assertIsNone(iv_pct)
        self.assertEqual(count, 0)

    def test_identical_values_in_history(self):
        history = {"AAPL": [
            {"date": "2026-01-01", "atm_iv": 0.25},
            {"date": "2026-01-02", "atm_iv": 0.25},
            {"date": "2026-01-03", "atm_iv": 0.25},
        ]}
        iv_rank, iv_pct, count = compute_iv_rank_percentile("AAPL", 0.25, history)
        self.assertIsNone(iv_rank)  # max == min
        self.assertEqual(count, 3)

    def test_current_at_max(self):
        history = {"AAPL": [
            {"date": "2026-01-01", "atm_iv": 0.10},
            {"date": "2026-01-02", "atm_iv": 0.50},
        ]}
        iv_rank, iv_pct, count = compute_iv_rank_percentile("AAPL", 0.50, history)
        self.assertAlmostEqual(iv_rank, 100.0)

    def test_current_at_min(self):
        history = {"AAPL": [
            {"date": "2026-01-01", "atm_iv": 0.10},
            {"date": "2026-01-02", "atm_iv": 0.50},
        ]}
        iv_rank, iv_pct, count = compute_iv_rank_percentile("AAPL", 0.10, history)
        self.assertAlmostEqual(iv_rank, 0.0)


class TestEstimateAtmIvEdgeCases(unittest.TestCase):
    def test_none_price(self):
        calls = pd.DataFrame({"strike": [100], "impliedVolatility": [0.25]})
        puts = pd.DataFrame({"strike": [100], "impliedVolatility": [0.27]})
        result = estimate_atm_iv(calls, puts, None)
        self.assertIsNone(result)

    def test_zero_price(self):
        calls = pd.DataFrame({"strike": [100], "impliedVolatility": [0.25]})
        puts = pd.DataFrame({"strike": [100], "impliedVolatility": [0.27]})
        result = estimate_atm_iv(calls, puts, 0)
        self.assertIsNone(result)

    def test_empty_dataframes(self):
        result = estimate_atm_iv(pd.DataFrame(), pd.DataFrame(), 100.0)
        self.assertIsNone(result)

    def test_only_calls(self):
        calls = pd.DataFrame({"strike": [100], "impliedVolatility": [0.25]})
        result = estimate_atm_iv(calls, pd.DataFrame(), 100.0)
        self.assertAlmostEqual(result, 0.25)

    def test_missing_columns(self):
        calls = pd.DataFrame({"price": [100]})  # wrong column names
        result = estimate_atm_iv(calls, pd.DataFrame(), 100.0)
        self.assertIsNone(result)


class TestAnnotateBlackScholesEdgeCases(unittest.TestCase):
    def test_empty_dataframe(self):
        result = annotate_black_scholes_greeks(pd.DataFrame(), "call", 100.0, "2026-06-01")
        self.assertEqual(len(result), 0)

    def test_none_dataframe(self):
        result = annotate_black_scholes_greeks(None, "call", 100.0, "2026-06-01")
        self.assertEqual(len(result), 0)

    def test_zero_underlying_price(self):
        chain = pd.DataFrame({"strike": [100], "impliedVolatility": [0.25], "contractSymbol": ["TEST"]})
        result = annotate_black_scholes_greeks(chain, "call", 0, "2026-06-01")
        # Should return copy without greeks
        self.assertNotIn("Delta", result.columns)

    def test_invalid_expiry(self):
        chain = pd.DataFrame({"strike": [100], "impliedVolatility": [0.25], "contractSymbol": ["TEST"]})
        result = annotate_black_scholes_greeks(chain, "call", 100.0, "invalid-date")
        self.assertNotIn("Delta", result.columns)

    def test_put_deltas_negative(self):
        expiry = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
        chain = pd.DataFrame({
            "contractSymbol": ["TESTP100"],
            "strike": [100],
            "impliedVolatility": [0.25],
        })
        result = annotate_black_scholes_greeks(chain, "put", 100.0, expiry)
        self.assertTrue(result["Delta"].iloc[0] < 0)

    def test_zero_iv_handled(self):
        expiry = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
        chain = pd.DataFrame({
            "contractSymbol": ["TESTC100"],
            "strike": [100],
            "impliedVolatility": [0],
        })
        result = annotate_black_scholes_greeks(chain, "call", 100.0, expiry)
        # Should still produce greeks with clamped IV
        self.assertIn("Delta", result.columns)


if __name__ == "__main__":
    unittest.main()
