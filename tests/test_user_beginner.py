import json
import unittest
from pathlib import Path

import pandas as pd

from sniper.options import normalize_market_symbol as _normalize_market_symbol
from sniper.persistence import load_watchlists, save_watchlists, get_watchlists_file_path


class _FakeTicker:
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


class TestBeginnerFlow(unittest.TestCase):
    def test_find_stock_and_add_watchlist(self):
        normalize = _normalize_market_symbol

        # Beginner: normalize ticker
        self.assertEqual(normalize(" aapl "), "AAPL")

        # Beginner: add a symbol to a watchlist and persist
        watchlists_data = {"MyList": ["aapl"]}
        save_watchlists(watchlists_data)
        loaded = load_watchlists()
        try:
            self.assertIn("MYLIST", {k.upper(): v for k, v in loaded.items()})
            self.assertIn("AAPL", [s.upper() for s in loaded.get("MyList", [])])
        finally:
            # cleanup persisted file
            p = Path(".streamlit") / "watchlists.json"
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    unittest.main()
