import unittest
from datetime import datetime, timedelta

import pandas as pd

from sniper.options import (
    estimate_atm_iv as _estimate_atm_iv,
    compute_iv_rank_percentile as _compute_iv_rank_percentile,
    annotate_black_scholes_greeks as _annotate_black_scholes_greeks,
)


class TestExpertFlow(unittest.TestCase):
    def test_estimate_iv_and_rank_and_greeks(self):
        history = {
            "AAPL": [
                {"date": "2026-01-01", "atm_iv": 0.20},
                {"date": "2026-01-02", "atm_iv": 0.30},
                {"date": "2026-01-03", "atm_iv": 0.40},
            ]
        }

        # ATM IV estimate
        calls = pd.DataFrame({"strike": [95, 100, 105], "impliedVolatility": [0.21, 0.24, 0.28]})
        puts = pd.DataFrame({"strike": [90, 100, 110], "impliedVolatility": [0.19, 0.26, 0.31]})

        atm = _estimate_atm_iv(calls, puts, 101)
        self.assertIsNotNone(atm)
        self.assertAlmostEqual(atm, (0.24 + 0.26) / 2, places=8)

        # IV rank and percentile
        iv_rank, iv_percentile, sample_size = _compute_iv_rank_percentile("aapl", 0.30, history)
        self.assertEqual(sample_size, 3)
        self.assertAlmostEqual(iv_rank, 50.0, places=6)
        self.assertAlmostEqual(iv_percentile, (2 / 3) * 100.0, places=6)

        # Black-Scholes greeks annotation
        expiry = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
        chain = pd.DataFrame({
            "contractSymbol": ["TESTC00100000", "TESTC00105000"],
            "strike": [100, 105],
            "impliedVolatility": [0.22, 0.25],
        })

        result = _annotate_black_scholes_greeks(chain, "call", 102.0, expiry)
        self.assertIn("Delta", result.columns)
        self.assertIn("Gamma", result.columns)
        self.assertIn("Theta/day", result.columns)
        self.assertIn("Vega", result.columns)
        self.assertEqual(len(result), 2)
        self.assertTrue(result["Delta"].notna().all())


if __name__ == "__main__":
    unittest.main()
