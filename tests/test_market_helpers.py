import ast
import math
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import pandas as pd


TARGET_FUNCTIONS = {
    "_normalize_market_symbol",
    "_compute_iv_rank_percentile",
    "_estimate_atm_iv",
    "_annotate_black_scholes_greeks",
}


def _load_market_helper_functions(options_iv_history):
    source_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in TARGET_FUNCTIONS:
            selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    compiled = compile(module, filename=str(source_path), mode="exec")

    fake_st = SimpleNamespace(session_state=SimpleNamespace(options_iv_history=options_iv_history))
    namespace = {
        "pd": pd,
        "math": math,
        "datetime": datetime,
        "NormalDist": NormalDist,
        "st": fake_st,
    }
    exec(compiled, namespace)
    return namespace


class TestMarketHelpers(unittest.TestCase):
    def test_estimate_atm_iv_averages_nearest_call_and_put(self):
        helpers = _load_market_helper_functions(options_iv_history={})
        estimate = helpers["_estimate_atm_iv"]

        calls = pd.DataFrame(
            {
                "strike": [95, 100, 105],
                "impliedVolatility": [0.21, 0.24, 0.28],
            }
        )
        puts = pd.DataFrame(
            {
                "strike": [90, 100, 110],
                "impliedVolatility": [0.19, 0.26, 0.31],
            }
        )

        result = estimate(calls, puts, 101)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, (0.24 + 0.26) / 2, places=8)

    def test_compute_iv_rank_percentile_with_history(self):
        history = {
            "AAPL": [
                {"date": "2026-01-01", "atm_iv": 0.20},
                {"date": "2026-01-02", "atm_iv": 0.30},
                {"date": "2026-01-03", "atm_iv": 0.40},
            ]
        }
        helpers = _load_market_helper_functions(options_iv_history=history)
        compute = helpers["_compute_iv_rank_percentile"]

        iv_rank, iv_percentile, sample_size = compute("aapl", 0.30)

        self.assertEqual(sample_size, 3)
        self.assertAlmostEqual(iv_rank, 50.0, places=6)
        self.assertAlmostEqual(iv_percentile, (2 / 3) * 100.0, places=6)

    def test_compute_iv_rank_percentile_requires_history(self):
        helpers = _load_market_helper_functions(options_iv_history={"MSFT": [{"date": "2026-01-01", "atm_iv": 0.25}]})
        compute = helpers["_compute_iv_rank_percentile"]

        iv_rank, iv_percentile, sample_size = compute("MSFT", 0.25)

        self.assertIsNone(iv_rank)
        self.assertIsNone(iv_percentile)
        self.assertEqual(sample_size, 1)

    def test_annotate_black_scholes_greeks_adds_columns(self):
        helpers = _load_market_helper_functions(options_iv_history={})
        annotate = helpers["_annotate_black_scholes_greeks"]

        expiry = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
        chain = pd.DataFrame(
            {
                "contractSymbol": ["TESTC00100000", "TESTC00105000"],
                "strike": [100, 105],
                "impliedVolatility": [0.22, 0.25],
            }
        )

        result = annotate(chain, "call", 102.0, expiry)

        self.assertIn("Delta", result.columns)
        self.assertIn("Gamma", result.columns)
        self.assertIn("Theta/day", result.columns)
        self.assertIn("Vega", result.columns)
        self.assertEqual(len(result), 2)
        self.assertTrue(result["Delta"].notna().all())


if __name__ == "__main__":
    unittest.main()
