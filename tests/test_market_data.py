"""Tests for sniper/market.py — pure functions (no API calls)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from sniper.market import (
    fetch_fundamental_snapshot,
    fetch_latest_close_prices,
    fetch_options_snapshot,
    fetch_quote_snapshot,
    get_base_symbol_universe,
    get_stock_data,
)


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


class TestFetchQuoteSnapshot(unittest.TestCase):
    """Tests for fetch_quote_snapshot."""

    def _make_ticker(self, info=None, fast_info=None):
        """Create a mock yf.Ticker with info and fast_info."""
        mock_ticker = MagicMock()
        mock_ticker.info = info if info is not None else {}
        mock_ticker.fast_info = fast_info if fast_info is not None else {}
        return mock_ticker

    @patch("sniper.market.yf.Ticker")
    def test_basic_quote(self, mock_yf_ticker):
        info = {
            "regularMarketPrice": 150.0,
            "regularMarketPreviousClose": 145.0,
            "marketCap": 2_500_000_000_000,
            "trailingPE": 28.5,
            "forwardPE": 25.0,
            "dividendYield": 0.006,
            "beta": 1.2,
            "fiftyTwoWeekHigh": 180.0,
            "fiftyTwoWeekLow": 120.0,
            "averageVolume": 50_000_000,
            "sector": "Technology",
        }
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        df = fetch_quote_snapshot(["AAPL"])
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["Symbol"], "AAPL")
        self.assertAlmostEqual(row["Price"], 150.0)
        expected_change = (150.0 - 145.0) / 145.0 * 100.0
        self.assertAlmostEqual(row["Change %"], expected_change, places=4)
        self.assertEqual(row["Market Cap"], 2_500_000_000_000)
        self.assertAlmostEqual(row["Div Yield %"], 0.6)
        self.assertEqual(row["Sector"], "Technology")

    @patch("sniper.market.yf.Ticker")
    def test_fallback_to_fast_info(self, mock_yf_ticker):
        """When info lacks price fields, fall back to fast_info."""
        info = {"sector": "Energy"}
        fast_info = {"lastPrice": 55.0, "previousClose": 50.0}
        mock_yf_ticker.return_value = self._make_ticker(info=info, fast_info=fast_info)
        df = fetch_quote_snapshot(["XOM"])
        row = df.iloc[0]
        self.assertAlmostEqual(row["Price"], 55.0)
        expected_change = (55.0 - 50.0) / 50.0 * 100.0
        self.assertAlmostEqual(row["Change %"], expected_change, places=4)

    @patch("sniper.market.yf.Ticker")
    def test_no_price_data(self, mock_yf_ticker):
        """When both info and fast_info lack prices."""
        mock_yf_ticker.return_value = self._make_ticker(info={}, fast_info={})
        df = fetch_quote_snapshot(["AAPL"])
        row = df.iloc[0]
        self.assertIsNone(row["Price"])
        self.assertIsNone(row["Change %"])

    @patch("sniper.market.yf.Ticker")
    def test_zero_current_price_no_change(self, mock_yf_ticker):
        """Zero current price should produce no change pct (guarded by not-in check)."""
        info = {"regularMarketPrice": 0, "regularMarketPreviousClose": 50.0}
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        df = fetch_quote_snapshot(["ZRO"])
        row = df.iloc[0]
        self.assertIsNone(row["Change %"])

    @patch("sniper.market.yf.Ticker")
    def test_zero_prev_close_no_change(self, mock_yf_ticker):
        """Zero prev_close should produce no change pct (avoids division by zero)."""
        info = {"regularMarketPrice": 100.0, "regularMarketPreviousClose": 0}
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        df = fetch_quote_snapshot(["TEST"])
        row = df.iloc[0]
        self.assertIsNone(row["Change %"])

    @patch("sniper.market.yf.Ticker")
    def test_multiple_symbols(self, mock_yf_ticker):
        info1 = {"regularMarketPrice": 150.0, "regularMarketPreviousClose": 145.0}
        info2 = {"regularMarketPrice": 300.0, "regularMarketPreviousClose": 310.0}
        ticker1 = self._make_ticker(info=info1)
        ticker2 = self._make_ticker(info=info2)
        mock_yf_ticker.side_effect = [ticker1, ticker2]
        df = fetch_quote_snapshot(["AAPL", "MSFT"])
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["Symbol"]), ["AAPL", "MSFT"])

    def test_empty_symbols_list(self):
        df = fetch_quote_snapshot([])
        self.assertTrue(df.empty)

    @patch("sniper.market.normalize_market_symbol", return_value="")
    def test_invalid_symbol_skipped(self, mock_norm):
        """Symbols that normalize to empty string are skipped."""
        df = fetch_quote_snapshot(["", " "])
        self.assertTrue(df.empty)
        self.assertEqual(mock_norm.call_count, 2)

    @patch("sniper.market.yf.Ticker")
    def test_exception_produces_partial_row(self, mock_yf_ticker):
        """If yf.Ticker raises, a row with only Symbol is appended."""
        mock_yf_ticker.side_effect = Exception("network error")
        df = fetch_quote_snapshot(["FAIL"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Symbol"], "FAIL")

    @patch("sniper.market.yf.Ticker")
    def test_ticker_info_not_dict(self, mock_yf_ticker):
        """If ticker.info is not a dict, treat as empty."""
        mock_ticker = MagicMock()
        mock_ticker.info = "not a dict"
        mock_ticker.fast_info = {"lastPrice": 42.0, "previousClose": 40.0}
        mock_yf_ticker.return_value = mock_ticker
        df = fetch_quote_snapshot(["ODD"])
        row = df.iloc[0]
        self.assertAlmostEqual(row["Price"], 42.0)

    @patch("sniper.market.yf.Ticker")
    def test_dividend_yield_none(self, mock_yf_ticker):
        """If dividendYield is None, Div Yield % should be None."""
        info = {"regularMarketPrice": 100.0, "dividendYield": None}
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        df = fetch_quote_snapshot(["NODIV"])
        self.assertIsNone(df.iloc[0]["Div Yield %"])

    @patch("sniper.market.yf.Ticker")
    def test_dividend_yield_present(self, mock_yf_ticker):
        """If dividendYield is present, Div Yield % = value * 100."""
        info = {"regularMarketPrice": 100.0, "dividendYield": 0.025}
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        df = fetch_quote_snapshot(["DIV"])
        self.assertAlmostEqual(df.iloc[0]["Div Yield %"], 2.5)

    @patch("sniper.market.yf.Ticker")
    @patch("sniper.market.normalize_market_symbol")
    def test_normalize_called_per_symbol(self, mock_norm, mock_yf_ticker):
        """Verify normalize_market_symbol is called for each symbol."""
        mock_norm.side_effect = lambda s: s.strip().upper() if s.strip() else ""
        mock_yf_ticker.return_value = self._make_ticker(
            info={"regularMarketPrice": 10.0}
        )
        fetch_quote_snapshot(["aapl", " msft "])
        self.assertEqual(mock_norm.call_count, 2)
        mock_norm.assert_any_call("aapl")
        mock_norm.assert_any_call(" msft ")


class TestFetchFundamentalSnapshot(unittest.TestCase):
    """Tests for fetch_fundamental_snapshot."""

    def _make_ticker(self, info=None, financials=None, balance_sheet=None,
                     cashflow=None, recommendations=None, earnings_dates=None):
        mock_ticker = MagicMock()
        mock_ticker.info = info if info is not None else {}
        mock_ticker.financials = financials if financials is not None else pd.DataFrame()
        mock_ticker.balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame()
        mock_ticker.cashflow = cashflow if cashflow is not None else pd.DataFrame()
        mock_ticker.recommendations = recommendations if recommendations is not None else pd.DataFrame()
        mock_ticker.earnings_dates = earnings_dates if earnings_dates is not None else pd.DataFrame()
        return mock_ticker

    @patch("sniper.market.normalize_market_symbol", return_value="")
    def test_empty_symbol_returns_empty_result(self, mock_norm):
        result = fetch_fundamental_snapshot("")
        self.assertEqual(result["metrics"], {})
        self.assertTrue(result["trend_df"].empty)
        self.assertTrue(result["income_df"].empty)
        self.assertEqual(result["factor_scores"], {})

    @patch("sniper.market.yf.Ticker")
    def test_basic_metrics(self, mock_yf_ticker):
        info = {
            "totalRevenue": 400_000_000_000,
            "netIncomeToCommon": 100_000_000_000,
            "trailingEps": 6.5,
            "grossMargins": 0.45,
            "operatingMargins": 0.30,
            "returnOnEquity": 0.15,
            "debtToEquity": 80.0,
            "earningsGrowth": 0.10,
        }
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        result = fetch_fundamental_snapshot("AAPL")
        m = result["metrics"]
        self.assertEqual(m["Revenue"], 400_000_000_000)
        self.assertEqual(m["Net Income"], 100_000_000_000)
        self.assertAlmostEqual(m["EPS"], 6.5)
        self.assertAlmostEqual(m["Gross Margin %"], 45.0)
        self.assertAlmostEqual(m["Operating Margin %"], 30.0)
        self.assertAlmostEqual(m["ROE %"], 15.0)
        self.assertAlmostEqual(m["Debt/Equity"], 80.0)
        self.assertAlmostEqual(m["Earnings Growth %"], 10.0)

    @patch("sniper.market.yf.Ticker")
    def test_analyst_snapshot(self, mock_yf_ticker):
        info = {
            "recommendationKey": "strong_buy",
            "targetMeanPrice": 200.0,
            "targetLowPrice": 150.0,
            "targetHighPrice": 250.0,
            "numberOfAnalystOpinions": 30,
        }
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        result = fetch_fundamental_snapshot("MSFT")
        snap = result["analyst_snapshot"]
        self.assertEqual(snap["Recommendation"], "Strong Buy")
        self.assertAlmostEqual(snap["Target Mean"], 200.0)
        self.assertAlmostEqual(snap["Target Low"], 150.0)
        self.assertAlmostEqual(snap["Target High"], 250.0)
        self.assertAlmostEqual(snap["Analyst Opinions"], 30.0)

    @patch("sniper.market.yf.Ticker")
    def test_factor_scores_momentum(self, mock_yf_ticker):
        info = {
            "regularMarketPrice": 160.0,
            "fiftyTwoWeekLow": 100.0,
            "fiftyTwoWeekHigh": 200.0,
        }
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        result = fetch_fundamental_snapshot("VRT")
        momentum = result["factor_scores"]["Momentum"]
        expected = ((160.0 - 100.0) / (200.0 - 100.0)) * 100.0
        self.assertAlmostEqual(momentum, expected)

    @patch("sniper.market.yf.Ticker")
    def test_momentum_none_when_no_range(self, mock_yf_ticker):
        """If 52w high == 52w low, momentum should be None (division guard)."""
        info = {
            "regularMarketPrice": 100.0,
            "fiftyTwoWeekLow": 100.0,
            "fiftyTwoWeekHigh": 100.0,
        }
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        result = fetch_fundamental_snapshot("FLAT")
        self.assertIsNone(result["factor_scores"]["Momentum"])

    @patch("sniper.market.yf.Ticker")
    def test_momentum_none_when_missing_price(self, mock_yf_ticker):
        info = {"fiftyTwoWeekLow": 50.0, "fiftyTwoWeekHigh": 150.0}
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        result = fetch_fundamental_snapshot("NOPRC")
        self.assertIsNone(result["factor_scores"]["Momentum"])

    @patch("sniper.market.yf.Ticker")
    def test_factor_scores_quality(self, mock_yf_ticker):
        info = {
            "grossMargins": 0.40,
            "operatingMargins": 0.20,
            "returnOnEquity": 0.18,
            "debtToEquity": 50.0,
        }
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        result = fetch_fundamental_snapshot("QUAL")
        quality = result["factor_scores"]["Quality"]
        self.assertIsNotNone(quality)
        self.assertGreater(quality, 0)
        self.assertLessEqual(quality, 100)

    @patch("sniper.market.yf.Ticker")
    def test_factor_scores_value(self, mock_yf_ticker):
        info = {
            "trailingPE": 15.0,
            "forwardPE": 12.0,
            "priceToBook": 2.0,
        }
        mock_yf_ticker.return_value = self._make_ticker(info=info)
        result = fetch_fundamental_snapshot("VAL")
        value = result["factor_scores"]["Value"]
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)

    @patch("sniper.market.yf.Ticker")
    def test_factor_scores_all_none_when_no_data(self, mock_yf_ticker):
        mock_yf_ticker.return_value = self._make_ticker(info={})
        result = fetch_fundamental_snapshot("EMPTY")
        fs = result["factor_scores"]
        self.assertIsNone(fs["Quality"])
        self.assertIsNone(fs["Growth"])
        self.assertIsNone(fs["Value"])
        self.assertIsNone(fs["Momentum"])

    @patch("sniper.market.yf.Ticker")
    def test_income_statement_trend_df(self, mock_yf_ticker):
        """Income statement with Total Revenue and Net Income produces trend_df."""
        dates = pd.to_datetime(["2023-12-31", "2022-12-31"])
        income_df = pd.DataFrame(
            {"Total Revenue": [200e9, 180e9], "Net Income": [50e9, 45e9]},
            index=dates,
        ).T
        income_df.columns = dates
        mock_yf_ticker.return_value = self._make_ticker(
            info={}, financials=income_df,
        )
        result = fetch_fundamental_snapshot("TRND")
        self.assertFalse(result["trend_df"].empty)
        self.assertIn("Total Revenue", result["trend_df"].columns)
        self.assertIn("Net Income", result["trend_df"].columns)

    @patch("sniper.market.yf.Ticker")
    def test_exception_returns_empty_result(self, mock_yf_ticker):
        mock_yf_ticker.side_effect = Exception("API error")
        result = fetch_fundamental_snapshot("FAIL")
        self.assertEqual(result["metrics"], {})
        self.assertTrue(result["trend_df"].empty)
        self.assertEqual(result["factor_scores"], {})

    @patch("sniper.market.yf.Ticker")
    def test_ticker_info_not_dict(self, mock_yf_ticker):
        mock_ticker = MagicMock()
        mock_ticker.info = "bad"
        mock_ticker.financials = pd.DataFrame()
        mock_ticker.balance_sheet = pd.DataFrame()
        mock_ticker.cashflow = pd.DataFrame()
        mock_ticker.recommendations = pd.DataFrame()
        mock_ticker.earnings_dates = pd.DataFrame()
        mock_yf_ticker.return_value = mock_ticker
        result = fetch_fundamental_snapshot("BAD")
        self.assertIsNone(result["metrics"]["Revenue"])

    @patch("sniper.market.yf.Ticker")
    def test_metrics_none_when_missing(self, mock_yf_ticker):
        mock_yf_ticker.return_value = self._make_ticker(info={})
        result = fetch_fundamental_snapshot("MISS")
        m = result["metrics"]
        self.assertIsNone(m["Revenue"])
        self.assertIsNone(m["Net Income"])
        self.assertIsNone(m["EPS"])
        self.assertIsNone(m["Gross Margin %"])

    @patch("sniper.market.yf.Ticker")
    def test_no_recommendation_key_returns_none(self, mock_yf_ticker):
        mock_yf_ticker.return_value = self._make_ticker(info={})
        result = fetch_fundamental_snapshot("NOREC")
        self.assertIsNone(result["analyst_snapshot"]["Recommendation"])

    @patch("sniper.market.yf.Ticker")
    def test_recommendations_datetime_index_reset(self, mock_yf_ticker):
        rec_df = pd.DataFrame(
            {"Firm": ["Goldman"], "To Grade": ["Buy"]},
            index=pd.to_datetime(["2024-01-15"]),
        )
        mock_yf_ticker.return_value = self._make_ticker(
            info={}, recommendations=rec_df,
        )
        result = fetch_fundamental_snapshot("REC")
        self.assertIn("Date", result["recommendations_df"].columns)

    @patch("sniper.market.yf.Ticker")
    def test_earnings_dates_datetime_index_reset(self, mock_yf_ticker):
        earn_df = pd.DataFrame(
            {"EPS Estimate": [2.5]},
            index=pd.to_datetime(["2024-04-25"]),
        )
        mock_yf_ticker.return_value = self._make_ticker(
            info={}, earnings_dates=earn_df,
        )
        result = fetch_fundamental_snapshot("EARN")
        self.assertIn("Date", result["earnings_dates_df"].columns)

    @patch("sniper.market.yf.Ticker")
    @patch("sniper.market.normalize_market_symbol")
    def test_normalize_called(self, mock_norm, mock_yf_ticker):
        mock_norm.return_value = "AAPL"
        mock_yf_ticker.return_value = self._make_ticker(info={})
        fetch_fundamental_snapshot("aapl")
        mock_norm.assert_called_once_with("aapl")

    @patch("sniper.market.yf.Ticker")
    def test_result_keys(self, mock_yf_ticker):
        mock_yf_ticker.return_value = self._make_ticker(info={})
        result = fetch_fundamental_snapshot("ANY")
        expected_keys = {
            "metrics", "trend_df", "income_df", "balance_df", "cashflow_df",
            "recommendations_df", "earnings_dates_df", "analyst_snapshot",
            "factor_scores",
        }
        self.assertEqual(set(result.keys()), expected_keys)


class TestFetchOptionsSnapshot(unittest.TestCase):
    """Tests for fetch_options_snapshot."""

    @patch("sniper.market.yf.Ticker")
    def test_basic_options_chain(self, mock_yf_ticker):
        mock_ticker = MagicMock()
        mock_ticker.options = ["2024-01-19", "2024-02-16"]
        calls_df = pd.DataFrame({"strike": [150, 155], "lastPrice": [5.0, 3.0]})
        puts_df = pd.DataFrame({"strike": [150, 145], "lastPrice": [2.0, 4.0]})
        mock_chain = SimpleNamespace(calls=calls_df, puts=puts_df)
        mock_ticker.option_chain.return_value = mock_chain
        mock_yf_ticker.return_value = mock_ticker

        expiries, calls, puts = fetch_options_snapshot("AAPL")
        self.assertEqual(expiries, ["2024-01-19", "2024-02-16"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(puts), 2)
        mock_ticker.option_chain.assert_called_once_with("2024-01-19")

    @patch("sniper.market.yf.Ticker")
    def test_specific_expiration(self, mock_yf_ticker):
        mock_ticker = MagicMock()
        mock_ticker.options = ["2024-01-19", "2024-02-16", "2024-03-15"]
        mock_chain = SimpleNamespace(
            calls=pd.DataFrame({"strike": [100]}),
            puts=pd.DataFrame({"strike": [100]}),
        )
        mock_ticker.option_chain.return_value = mock_chain
        mock_yf_ticker.return_value = mock_ticker

        expiries, calls, puts = fetch_options_snapshot("AAPL", expiration="2024-02-16")
        mock_ticker.option_chain.assert_called_once_with("2024-02-16")
        self.assertEqual(len(expiries), 3)

    @patch("sniper.market.yf.Ticker")
    def test_invalid_expiration_falls_back(self, mock_yf_ticker):
        """If requested expiration not in list, use first available."""
        mock_ticker = MagicMock()
        mock_ticker.options = ["2024-01-19", "2024-02-16"]
        mock_chain = SimpleNamespace(
            calls=pd.DataFrame(), puts=pd.DataFrame(),
        )
        mock_ticker.option_chain.return_value = mock_chain
        mock_yf_ticker.return_value = mock_ticker

        fetch_options_snapshot("AAPL", expiration="2099-12-31")
        mock_ticker.option_chain.assert_called_once_with("2024-01-19")

    @patch("sniper.market.yf.Ticker")
    def test_no_expiries_returns_empty(self, mock_yf_ticker):
        mock_ticker = MagicMock()
        mock_ticker.options = []
        mock_yf_ticker.return_value = mock_ticker
        expiries, calls, puts = fetch_options_snapshot("NOOPT")
        self.assertEqual(expiries, [])
        self.assertTrue(calls.empty)
        self.assertTrue(puts.empty)

    @patch("sniper.market.yf.Ticker")
    def test_options_attr_none(self, mock_yf_ticker):
        mock_ticker = MagicMock()
        mock_ticker.options = None
        mock_yf_ticker.return_value = mock_ticker
        expiries, calls, puts = fetch_options_snapshot("NONEOPT")
        self.assertEqual(expiries, [])
        self.assertTrue(calls.empty)
        self.assertTrue(puts.empty)

    @patch("sniper.market.normalize_market_symbol", return_value="")
    def test_empty_symbol_returns_empty(self, mock_norm):
        expiries, calls, puts = fetch_options_snapshot("")
        self.assertEqual(expiries, [])
        self.assertTrue(calls.empty)
        self.assertTrue(puts.empty)

    @patch("sniper.market.yf.Ticker")
    def test_exception_returns_empty(self, mock_yf_ticker):
        mock_yf_ticker.side_effect = Exception("API down")
        expiries, calls, puts = fetch_options_snapshot("FAIL")
        self.assertEqual(expiries, [])
        self.assertTrue(calls.empty)
        self.assertTrue(puts.empty)

    @patch("sniper.market.yf.Ticker")
    def test_chain_missing_calls_attr(self, mock_yf_ticker):
        """If chain lacks 'calls' attribute, return empty DataFrame."""
        mock_ticker = MagicMock()
        mock_ticker.options = ["2024-01-19"]
        mock_chain = SimpleNamespace(puts=pd.DataFrame({"strike": [100]}))
        mock_ticker.option_chain.return_value = mock_chain
        mock_yf_ticker.return_value = mock_ticker
        expiries, calls, puts = fetch_options_snapshot("AAPL")
        self.assertTrue(calls.empty)
        self.assertEqual(len(puts), 1)

    @patch("sniper.market.yf.Ticker")
    def test_chain_missing_puts_attr(self, mock_yf_ticker):
        mock_ticker = MagicMock()
        mock_ticker.options = ["2024-01-19"]
        mock_chain = SimpleNamespace(calls=pd.DataFrame({"strike": [100]}))
        mock_ticker.option_chain.return_value = mock_chain
        mock_yf_ticker.return_value = mock_ticker
        expiries, calls, puts = fetch_options_snapshot("AAPL")
        self.assertEqual(len(calls), 1)
        self.assertTrue(puts.empty)

    @patch("sniper.market.yf.Ticker")
    @patch("sniper.market.normalize_market_symbol")
    def test_normalize_called(self, mock_norm, mock_yf_ticker):
        mock_norm.return_value = "AAPL"
        mock_ticker = MagicMock()
        mock_ticker.options = []
        mock_yf_ticker.return_value = mock_ticker
        fetch_options_snapshot("aapl")
        mock_norm.assert_called_once_with("aapl")


class TestFetchLatestClosePrices(unittest.TestCase):
    """Tests for fetch_latest_close_prices."""

    def test_empty_tickers_returns_empty(self):
        result = fetch_latest_close_prices(())
        self.assertEqual(result, {})

    @patch("sniper.market.yf.download")
    def test_single_ticker(self, mock_download):
        """Single ticker: yf.download returns Series for Close column."""
        close_series = pd.Series([152.5], index=pd.to_datetime(["2024-01-15"]))
        mock_download.return_value = pd.DataFrame({"Close": close_series, "Open": [150.0]})
        result = fetch_latest_close_prices(("AAPL",))
        self.assertAlmostEqual(result["AAPL"], 152.5)

    @patch("sniper.market.yf.download")
    def test_multiple_tickers(self, mock_download):
        close_df = pd.DataFrame(
            {"AAPL": [152.5], "MSFT": [420.0]},
            index=pd.to_datetime(["2024-01-15"]),
        )
        full_df = pd.DataFrame({
            ("Close", "AAPL"): [152.5],
            ("Close", "MSFT"): [420.0],
            ("Open", "AAPL"): [150.0],
            ("Open", "MSFT"): [418.0],
        }, index=pd.to_datetime(["2024-01-15"]))
        full_df.columns = pd.MultiIndex.from_tuples(full_df.columns)
        mock_download.return_value = full_df
        result = fetch_latest_close_prices(("AAPL", "MSFT"))
        self.assertAlmostEqual(result["AAPL"], 152.5)
        self.assertAlmostEqual(result["MSFT"], 420.0)

    @patch("sniper.market.yf.download")
    def test_download_returns_none(self, mock_download):
        mock_download.return_value = None
        result = fetch_latest_close_prices(("AAPL", "MSFT"))
        self.assertEqual(result, {"AAPL": 0.0, "MSFT": 0.0})

    @patch("sniper.market.yf.download")
    def test_download_returns_empty_df(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        result = fetch_latest_close_prices(("AAPL",))
        self.assertEqual(result, {"AAPL": 0.0})

    @patch("sniper.market.yf.download")
    def test_no_close_column(self, mock_download):
        """DataFrame without 'Close' column."""
        mock_download.return_value = pd.DataFrame(
            {"Open": [150.0]}, index=pd.to_datetime(["2024-01-15"]),
        )
        result = fetch_latest_close_prices(("AAPL",))
        self.assertEqual(result, {"AAPL": 0.0})

    @patch("sniper.market.yf.download")
    def test_exception_returns_zeros(self, mock_download):
        mock_download.side_effect = Exception("network error")
        result = fetch_latest_close_prices(("AAPL", "MSFT"))
        self.assertEqual(result, {"AAPL": 0.0, "MSFT": 0.0})

    @patch("sniper.market.yf.download")
    def test_missing_ticker_in_close_defaults_to_zero(self, mock_download):
        """If a ticker is missing from the Close frame, default to 0.0."""
        close_df = pd.DataFrame(
            {"AAPL": [152.5]},
            index=pd.to_datetime(["2024-01-15"]),
        )
        full_df = pd.DataFrame({
            ("Close", "AAPL"): [152.5],
        }, index=pd.to_datetime(["2024-01-15"]))
        full_df.columns = pd.MultiIndex.from_tuples(full_df.columns)
        mock_download.return_value = full_df
        result = fetch_latest_close_prices(("AAPL", "MISSING"))
        self.assertAlmostEqual(result["AAPL"], 152.5)
        self.assertAlmostEqual(result["MISSING"], 0.0)

    @patch("sniper.market.yf.download")
    def test_nan_value_converts_to_float(self, mock_download):
        """NaN close price is converted via float(); verify key exists with a float value."""
        full_df = pd.DataFrame({
            ("Close", "AAPL"): [float("nan")],
        }, index=pd.to_datetime(["2024-01-15"]))
        full_df.columns = pd.MultiIndex.from_tuples(full_df.columns)
        mock_download.return_value = full_df
        result = fetch_latest_close_prices(("AAPL",))
        self.assertIn("AAPL", result)
        # float(NaN) is valid in Python — the function converts without error
        self.assertIsInstance(result["AAPL"], float)

    @patch("sniper.market.yf.download")
    def test_close_is_series_multi_tickers_returns_zeros(self, mock_download):
        """If Close is a Series but multiple tickers requested, return zeros."""
        df = pd.DataFrame({"Close": [150.0]}, index=pd.to_datetime(["2024-01-15"]))
        mock_download.return_value = df
        result = fetch_latest_close_prices(("AAPL", "MSFT"))
        self.assertEqual(result, {"AAPL": 0.0, "MSFT": 0.0})

    @patch("sniper.market.yf.download")
    def test_download_called_with_correct_args(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        fetch_latest_close_prices(("AAPL", "MSFT"))
        mock_download.assert_called_once_with(("AAPL", "MSFT"), period="1d", progress=False)


if __name__ == "__main__":
    unittest.main()
