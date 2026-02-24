import ast
import json
import unittest
from pathlib import Path

import pandas as pd


TARGET_FUNCTIONS = {
    "_normalize_market_symbol",
    "fetch_quote_snapshot",
    "load_watchlists",
    "save_watchlists",
}


def _load_functions():
    source_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in TARGET_FUNCTIONS:
            # strip decorators (e.g., @st.cache_data) to avoid evaluation at exec time
            node.decorator_list = []
            selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    compiled = compile(module, filename=str(source_path), mode="exec")

    # Fake minimal yf module for deterministic quote snapshot
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.info = {
                "regularMarketPrice": 150.0,
                "regularMarketPreviousClose": 148.0,
                "marketCap": 1_000_000_000,
                "trailingPE": 25.0,
                "forwardPE": 23.0,
                "dividendYield": 0.01,
                "beta": 1.1,
                "fiftyTwoWeekHigh": 182.0,
                "fiftyTwoWeekLow": 120.0,
                "averageVolume": 10_000_000,
                "sector": "Technology",
            }
            self.fast_info = {"lastPrice": 150.0, "previousClose": 148.0}

    fake_yf = type("yf", (), {"Ticker": FakeTicker})

    # Minimal fake `st` namespace used by load/save watchlists
    class FakeSessionState:
        def __init__(self):
            self.watchlists = {}

    class FakeSt:
        def __init__(self):
            self.session_state = FakeSessionState()

    fake_st = FakeSt()

    namespace = {
        "pd": pd,
        "yf": fake_yf,
        "st": fake_st,
        "Path": Path,
        "json": json,
    }

    exec(compiled, namespace)
    return namespace, fake_st


class TestBeginnerFlow(unittest.TestCase):
    def test_find_stock_and_add_watchlist(self):
        ns, fake_st = _load_functions()

        fetch = ns["fetch_quote_snapshot"]
        normalize = ns["_normalize_market_symbol"]
        save_watchlists = ns["save_watchlists"]
        load_watchlists = ns["load_watchlists"]

        # Beginner: normalize ticker
        self.assertEqual(normalize(" aapl "), "AAPL")

        # Beginner: fetch quote snapshot for single symbol using fake yf
        df = fetch(["AAPL"])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Symbol", df.columns)
        self.assertEqual(df.iloc[0]["Symbol"], "AAPL")
        self.assertAlmostEqual(df.iloc[0]["Price"], 150.0)

        # Beginner: add a symbol to a watchlist in session_state
        # save_watchlists is a no-op on cloud; state lives in session_state
        fake_st.session_state.watchlists = {"MyList": ["AAPL"]}
        save_watchlists()  # no-op — confirms it doesn't raise
        self.assertIn("MyList", fake_st.session_state.watchlists)
        self.assertIn("AAPL", fake_st.session_state.watchlists["MyList"])

        # load_watchlists returns empty dict (cloud session-state only)
        loaded = load_watchlists()
        self.assertIsInstance(loaded, dict)


if __name__ == "__main__":
    unittest.main()
