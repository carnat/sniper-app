import ast
import math
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import pandas as pd


TARGET_FUNCTIONS = {
    "_compute_iv_rank_percentile",
    "_estimate_atm_iv",
    "_annotate_black_scholes_greeks",
    "_normalize_market_symbol",
}


def _load_market_helper_functions(options_iv_history=None):
    source_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in TARGET_FUNCTIONS:
            selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    compiled = compile(module, filename=str(source_path), mode="exec")

    fake_st = type("st", (), {"session_state": type("ss", (), {"options_iv_history": options_iv_history or {}})})

    namespace = {
        "pd": pd,
        "math": math,
        "datetime": datetime,
        "NormalDist": NormalDist,
        "st": fake_st,
    }
    exec(compiled, namespace)
    return namespace


class TestExpertFlow(unittest.TestCase):
    def test_estimate_iv_and_rank_and_greeks(self):
        helpers = _load_market_helper_functions(
            options_iv_history={
                "AAPL": [
                    {"date": "2026-01-01", "atm_iv": 0.20},
                    {"date": "2026-01-02", "atm_iv": 0.30},
                    {"date": "2026-01-03", "atm_iv": 0.40},
                ]
            }
        )

        estimate = helpers["_estimate_atm_iv"]
        compute = helpers["_compute_iv_rank_percentile"]
        annotate = helpers["_annotate_black_scholes_greeks"]

        calls = pd.DataFrame({"strike": [95, 100, 105], "impliedVolatility": [0.21, 0.24, 0.28]})
        puts = pd.DataFrame({"strike": [90, 100, 110], "impliedVolatility": [0.19, 0.26, 0.31]})

        # ATM IV estimate
        atm = estimate(calls, puts, 101)
        self.assertIsNotNone(atm)
        self.assertAlmostEqual(atm, (0.24 + 0.26) / 2, places=8)

        # IV rank and percentile
        iv_rank, iv_percentile, sample_size = compute("aapl", 0.30)
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

        result = annotate(chain, "call", 102.0, expiry)
        self.assertIn("Delta", result.columns)
        self.assertIn("Gamma", result.columns)
        self.assertIn("Theta/day", result.columns)
        self.assertIn("Vega", result.columns)
        self.assertEqual(len(result), 2)
        self.assertTrue(result["Delta"].notna().all())


if __name__ == "__main__":
    unittest.main()
