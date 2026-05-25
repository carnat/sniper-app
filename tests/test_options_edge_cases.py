"""Additional edge-case tests for sniper.options."""

import unittest
from datetime import datetime, timedelta

import pandas as pd

from sniper.options import (
    annotate_black_scholes_greeks,
    compute_iv_rank_percentile,
    estimate_atm_iv,
    record_atm_iv,
)


class TestEstimateAtmIvEdgeCases(unittest.TestCase):
    def test_estimate_atm_iv_returns_none_for_zero_underlying_price(self):
        self.assertIsNone(estimate_atm_iv(pd.DataFrame(), pd.DataFrame(), 0))

    def test_estimate_atm_iv_returns_none_for_none_underlying_price(self):
        self.assertIsNone(estimate_atm_iv(pd.DataFrame(), pd.DataFrame(), None))

    def test_estimate_atm_iv_returns_none_for_empty_dataframes(self):
        self.assertIsNone(estimate_atm_iv(pd.DataFrame(), pd.DataFrame(), 100.0))

    def test_estimate_atm_iv_returns_none_when_all_implied_volatilities_are_nan(self):
        calls = pd.DataFrame({"strike": [95, 100], "impliedVolatility": [float("nan"), float("nan")]})
        puts = pd.DataFrame({"strike": [100], "impliedVolatility": [float("nan")]})

        self.assertIsNone(estimate_atm_iv(calls, puts, 100.0))

    def test_estimate_atm_iv_returns_none_when_required_columns_are_missing(self):
        calls = pd.DataFrame({"price": [1.0]})
        puts = pd.DataFrame({"strike": [100], "iv": [0.3]})

        self.assertIsNone(estimate_atm_iv(calls, puts, 100.0))

    def test_estimate_atm_iv_happy_path_averages_nearest_call_and_put(self):
        calls = pd.DataFrame({"strike": [95, 101, 110], "impliedVolatility": [0.2, 0.3, 0.4]})
        puts = pd.DataFrame({"strike": [90, 99, 120], "impliedVolatility": [0.1, 0.4, 0.5]})

        result = estimate_atm_iv(calls, puts, 100.0)

        self.assertAlmostEqual(result, 0.35)


class TestAnnotateBlackScholesGreeksEdgeCases(unittest.TestCase):
    def test_annotate_black_scholes_greeks_with_empty_dataframe_returns_empty_dataframe(self):
        result = annotate_black_scholes_greeks(pd.DataFrame(), "call", 100.0, "2026-06-01")

        self.assertTrue(result.empty)

    def test_annotate_black_scholes_greeks_with_zero_underlying_price_returns_copy_without_greeks(self):
        chain = pd.DataFrame({"strike": [100], "impliedVolatility": [0.25]})

        result = annotate_black_scholes_greeks(chain, "call", 0, "2026-06-01")

        self.assertIsNot(result, chain)
        self.assertListEqual(list(result.columns), list(chain.columns))

    def test_annotate_black_scholes_greeks_with_invalid_expiry_returns_copy(self):
        chain = pd.DataFrame({"strike": [100], "impliedVolatility": [0.25]})

        result = annotate_black_scholes_greeks(chain, "call", 100.0, "not-a-date")

        self.assertIsNot(result, chain)
        self.assertListEqual(list(result.columns), list(chain.columns))

    def test_annotate_black_scholes_greeks_sets_none_for_zero_strike_or_zero_sigma(self):
        expiry = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
        chain = pd.DataFrame(
            {
                "strike": [0, 100, 100],
                "impliedVolatility": [0.25, 0.0, 0.25],
            }
        )

        result = annotate_black_scholes_greeks(chain, "call", 100.0, expiry)

        self.assertTrue(pd.isna(result.loc[0, "Delta"]))
        self.assertTrue(pd.isna(result.loc[1, "Delta"]))
        self.assertFalse(pd.isna(result.loc[2, "Delta"]))

    def test_annotate_black_scholes_greeks_call_and_put_delta_have_expected_signs(self):
        expiry = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
        chain = pd.DataFrame({"strike": [100], "impliedVolatility": [0.25]})

        call_result = annotate_black_scholes_greeks(chain, "call", 100.0, expiry)
        put_result = annotate_black_scholes_greeks(chain, "put", 100.0, expiry)

        self.assertGreater(call_result.loc[0, "Delta"], 0)
        self.assertLess(put_result.loc[0, "Delta"], 0)


class TestComputeIvRankPercentileEdgeCases(unittest.TestCase):
    def test_compute_iv_rank_percentile_with_none_current_iv(self):
        self.assertEqual(compute_iv_rank_percentile("AAPL", None, {}), (None, None, 0))

    def test_compute_iv_rank_percentile_with_less_than_two_history_values(self):
        history = {"AAPL": [{"date": "2026-01-01", "atm_iv": 0.2}]}

        self.assertEqual(compute_iv_rank_percentile("AAPL", 0.2, history), (None, None, 1))

    def test_compute_iv_rank_percentile_with_equal_min_and_max_has_no_rank(self):
        history = {"AAPL": [{"date": "2026-01-01", "atm_iv": 0.2}, {"date": "2026-01-02", "atm_iv": 0.2}]}

        iv_rank, iv_percentile, count = compute_iv_rank_percentile("AAPL", 0.2, history)

        self.assertIsNone(iv_rank)
        self.assertEqual(iv_percentile, 100.0)
        self.assertEqual(count, 2)

    def test_compute_iv_rank_percentile_normal_case(self):
        history = {
            "AAPL": [
                {"date": "2026-01-01", "atm_iv": 0.1},
                {"date": "2026-01-02", "atm_iv": 0.2},
                {"date": "2026-01-03", "atm_iv": 0.3},
            ]
        }

        iv_rank, iv_percentile, count = compute_iv_rank_percentile("AAPL", 0.2, history)

        self.assertAlmostEqual(iv_rank, 50.0)
        self.assertAlmostEqual(iv_percentile, 66.66666666666666)
        self.assertEqual(count, 3)


class TestRecordAtmIvEdgeCases(unittest.TestCase):
    def test_record_atm_iv_with_none_value_makes_no_change(self):
        history = {"AAPL": [{"date": "2026-01-01", "atm_iv": 0.2}]}

        result = record_atm_iv("AAPL", None, history)

        self.assertEqual(result, history)

    def test_record_atm_iv_with_empty_symbol_makes_no_change(self):
        history = {"AAPL": [{"date": "2026-01-01", "atm_iv": 0.2}]}

        result = record_atm_iv("", 0.3, history)

        self.assertEqual(result, history)

    def test_record_atm_iv_replaces_same_day_entry_instead_of_duplicating(self):
        today = datetime.now().strftime("%Y-%m-%d")
        history = {"AAPL": [{"date": today, "atm_iv": 0.2}]}

        result = record_atm_iv("AAPL", 0.35, history)

        self.assertEqual(len(result["AAPL"]), 1)
        self.assertEqual(result["AAPL"][0]["date"], today)
        self.assertAlmostEqual(result["AAPL"][0]["atm_iv"], 0.35)


if __name__ == "__main__":
    unittest.main()
