"""Tests for sniper/news.py — pure helper functions (no API calls)."""

import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from sniper.news import (
    get_finnhub_api_key,
    _extract_domain,
    format_news_timestamp,
    resolve_symbol_with_search,
    get_watchtower_market_snapshot,
    fetch_news_for_ticker,
    get_earnings_dates,
)


class TestGetFinnhubApiKey(unittest.TestCase):
    @patch.dict(os.environ, {"FINNHUB_API_KEY": ""}, clear=False)
    def test_from_secrets_key(self):
        result = get_finnhub_api_key(secrets_key="real_api_key_123")
        self.assertEqual(result, "real_api_key_123")

    @patch.dict(os.environ, {"FINNHUB_API_KEY": "env_key_abc"}, clear=False)
    def test_from_env(self):
        result = get_finnhub_api_key(secrets_key=None)
        self.assertEqual(result, "env_key_abc")

    @patch.dict(os.environ, {"FINNHUB_API_KEY": ""}, clear=False)
    def test_filters_placeholder(self):
        result = get_finnhub_api_key(secrets_key="your_finnhub_api_key_here")
        self.assertEqual(result, "")

    @patch.dict(os.environ, {"FINNHUB_API_KEY": ""}, clear=False)
    def test_filters_placeholder_variants(self):
        for placeholder in ["YOUR_API_KEY", "placeholder_key", "FINNHUB_API_KEY_HERE"]:
            result = get_finnhub_api_key(secrets_key=placeholder)
            self.assertEqual(result, "", f"Should filter: {placeholder}")

    @patch.dict(os.environ, {"FINNHUB_API_KEY": ""}, clear=False)
    def test_returns_empty_when_none(self):
        result = get_finnhub_api_key(secrets_key=None)
        self.assertEqual(result, "")

    @patch.dict(os.environ, {"FINNHUB_API_KEY": "valid_env_key"}, clear=False)
    def test_secrets_key_takes_priority(self):
        result = get_finnhub_api_key(secrets_key="valid_secrets_key")
        self.assertEqual(result, "valid_secrets_key")


class TestExtractDomain(unittest.TestCase):
    def test_basic_url(self):
        self.assertEqual(_extract_domain("https://example.com/path"), "example.com")

    def test_www_stripped(self):
        self.assertEqual(_extract_domain("https://www.reuters.com/article"), "reuters.com")

    def test_http_url(self):
        self.assertEqual(_extract_domain("http://finance.yahoo.com/news"), "finance.yahoo.com")

    def test_empty_url(self):
        self.assertEqual(_extract_domain(""), "")

    def test_none_url(self):
        self.assertEqual(_extract_domain(None), "")

    def test_url_with_port(self):
        result = _extract_domain("https://localhost:8080/path")
        self.assertEqual(result, "localhost:8080")


class TestFormatNewsTimestamp(unittest.TestCase):
    def test_recent_timestamp(self):
        # Use a timestamp from right now
        import pandas as pd
        now = pd.Timestamp.now(tz="UTC")
        relative, exact = format_news_timestamp(now.isoformat())
        self.assertEqual(relative, "just now")
        self.assertNotEqual(exact, "Unknown")

    def test_hours_ago(self):
        import pandas as pd
        ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3)).isoformat()
        relative, exact = format_news_timestamp(ts)
        self.assertIn("h ago", relative)

    def test_days_ago(self):
        import pandas as pd
        ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5)).isoformat()
        relative, exact = format_news_timestamp(ts)
        self.assertIn("d ago", relative)

    def test_invalid_timestamp(self):
        relative, exact = format_news_timestamp("not-a-date")
        self.assertEqual(relative, "Unknown")
        self.assertEqual(exact, "Unknown")

    def test_none_input(self):
        relative, exact = format_news_timestamp(None)
        self.assertEqual(relative, "Unknown")


class TestResolveSymbolWithSearch(unittest.TestCase):
    """Tests for resolve_symbol_with_search — yfinance Search symbol resolution."""

    @patch("sniper.news.yf.Search")
    def test_basic_resolution(self, mock_search):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple Inc.", "longname": "Apple Inc.", "exchange": "NMS"},
        ]
        result = resolve_symbol_with_search("AAPL")
        self.assertEqual(result["input"], "AAPL")
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["shortname"], "Apple Inc.")
        self.assertEqual(result["longname"], "Apple Inc.")
        self.assertEqual(result["exchange"], "NMS")

    def test_empty_ticker(self):
        result = resolve_symbol_with_search("")
        self.assertEqual(result, {"input": "", "symbol": "", "shortname": "", "longname": "", "exchange": ""})

    def test_none_ticker(self):
        result = resolve_symbol_with_search(None)
        self.assertEqual(result["input"], "")
        self.assertEqual(result["symbol"], "")

    @patch("sniper.news.yf.Search")
    def test_no_quotes_returns_fallback(self, mock_search):
        mock_search.return_value.quotes = []
        result = resolve_symbol_with_search("XYZ")
        self.assertEqual(result["input"], "XYZ")
        self.assertEqual(result["symbol"], "XYZ")
        self.assertEqual(result["shortname"], "")

    @patch("sniper.news.yf.Search")
    def test_no_quotes_bk_ticker(self, mock_search):
        mock_search.return_value.quotes = []
        result = resolve_symbol_with_search("PTT.BK")
        self.assertEqual(result["symbol"], "PTT.BK")

    @patch("sniper.news.yf.Search")
    def test_none_quotes_attribute(self, mock_search):
        mock_search.return_value.quotes = None
        result = resolve_symbol_with_search("MSFT")
        self.assertEqual(result["symbol"], "MSFT")

    @patch("sniper.news.yf.Search")
    def test_scoring_exact_match_wins(self, mock_search):
        mock_search.return_value.quotes = [
            {"symbol": "VRT.AX", "shortname": "Vertiv AX", "longname": "", "exchange": "ASX"},
            {"symbol": "VRT", "shortname": "Vertiv", "longname": "Vertiv Holdings", "exchange": "NYSE"},
        ]
        result = resolve_symbol_with_search("VRT")
        self.assertEqual(result["symbol"], "VRT")
        self.assertEqual(result["shortname"], "Vertiv")

    @patch("sniper.news.yf.Search")
    def test_scoring_bk_exchange_preference(self, mock_search):
        mock_search.return_value.quotes = [
            {"symbol": "PTT", "shortname": "PTT US", "longname": "", "exchange": "NYSE"},
            {"symbol": "PTT.BK", "shortname": "PTT PCL", "longname": "PTT PCL", "exchange": "SET"},
        ]
        result = resolve_symbol_with_search("PTT.BK")
        self.assertEqual(result["symbol"], "PTT.BK")

    @patch("sniper.news.yf.Search")
    def test_scoring_bk_bkk_exchange(self, mock_search):
        mock_search.return_value.quotes = [
            {"symbol": "AOT.BK", "shortname": "Airports of TH", "longname": "", "exchange": "BKK"},
        ]
        result = resolve_symbol_with_search("AOT.BK")
        self.assertEqual(result["symbol"], "AOT.BK")

    @patch("sniper.news.yf.Search")
    def test_search_exception_returns_fallback(self, mock_search):
        mock_search.side_effect = Exception("network error")
        result = resolve_symbol_with_search("AAPL")
        self.assertEqual(result["input"], "AAPL")
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["shortname"], "")

    @patch("sniper.news.yf.Search")
    def test_missing_fields_in_quote(self, mock_search):
        mock_search.return_value.quotes = [{"symbol": "TSM"}]
        result = resolve_symbol_with_search("TSM")
        self.assertEqual(result["symbol"], "TSM")
        self.assertEqual(result["shortname"], "")
        self.assertEqual(result["longname"], "")
        self.assertEqual(result["exchange"], "")

    @patch("sniper.news.yf.Search")
    def test_none_values_in_quote(self, mock_search):
        mock_search.return_value.quotes = [
            {"symbol": None, "shortname": None, "longname": None, "exchange": None},
        ]
        result = resolve_symbol_with_search("AAPL")
        # symbol falls back to raw when quote symbol is empty
        self.assertEqual(result["symbol"], "AAPL")

    @patch("sniper.news.yf.Search")
    def test_whitespace_ticker_stripped(self, mock_search):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"},
        ]
        result = resolve_symbol_with_search("  aapl  ")
        self.assertEqual(result["input"], "AAPL")
        self.assertEqual(result["symbol"], "AAPL")

    @patch("sniper.news.yf.Search")
    def test_best_match_with_stripped_bk(self, mock_search):
        """A quote whose symbol.replace('.BK','') matches stripped gets +80."""
        mock_search.return_value.quotes = [
            {"symbol": "SCC.BK", "shortname": "SCG", "longname": "", "exchange": "SET"},
        ]
        result = resolve_symbol_with_search("SCC")
        self.assertEqual(result["symbol"], "SCC.BK")


class TestGetWatchtowerMarketSnapshot(unittest.TestCase):
    """Tests for get_watchtower_market_snapshot — market status + index data."""

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_full_snapshot(self, mock_market_cls, mock_download):
        mock_market = MagicMock()
        mock_market.status = {"status": "Open", "message": "Regular hours"}
        mock_market_cls.return_value = mock_market

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
        close_df = pd.DataFrame(
            {"^GSPC": [4800.0, 4850.0], "^IXIC": [15000.0, 15100.0],
             "^DJI": [37000.0, 37200.0], "^SET.BK": [1400.0, 1410.0]},
            index=idx,
        )
        hist = pd.DataFrame({"Close": [0]})  # dummy top-level
        hist.get = lambda key, default=None: close_df if key == "Close" else default
        mock_download.return_value = hist

        result = get_watchtower_market_snapshot()
        self.assertEqual(result["us_status"], "Open")
        self.assertEqual(result["us_message"], "Regular hours")
        self.assertIn("^GSPC", result["indices"])
        self.assertAlmostEqual(result["indices"]["^GSPC"]["last"], 4850.0)
        expected_pct = (4850.0 - 4800.0) / 4800.0 * 100.0
        self.assertAlmostEqual(result["indices"]["^GSPC"]["pct"], expected_pct, places=4)

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_market_exception_still_returns_snapshot(self, mock_market_cls, mock_download):
        mock_market_cls.side_effect = Exception("API down")
        mock_download.return_value = pd.DataFrame()

        result = get_watchtower_market_snapshot()
        self.assertEqual(result["us_status"], "Unknown")
        self.assertEqual(result["us_message"], "")
        self.assertEqual(result["indices"], {})

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_download_exception_still_returns_snapshot(self, mock_market_cls, mock_download):
        mock_market = MagicMock()
        mock_market.status = {"status": "Closed", "message": "After hours"}
        mock_market_cls.return_value = mock_market
        mock_download.side_effect = Exception("download failed")

        result = get_watchtower_market_snapshot()
        self.assertEqual(result["us_status"], "Closed")
        self.assertEqual(result["indices"], {})

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_non_dict_status(self, mock_market_cls, mock_download):
        mock_market = MagicMock()
        mock_market.status = "string_status"
        mock_market_cls.return_value = mock_market
        mock_download.return_value = pd.DataFrame()

        result = get_watchtower_market_snapshot()
        self.assertEqual(result["us_status"], "Unknown")

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_close_as_series_branch(self, mock_market_cls, mock_download):
        """When hist.get('Close') returns a Series (single symbol case)."""
        mock_market_cls.return_value = MagicMock(status={})

        close_series = pd.Series([5000.0, 5100.0], index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]))
        hist = pd.DataFrame({"Close": [0]})
        hist.get = lambda key, default=None: close_series if key == "Close" else default

        mock_download.return_value = hist

        result = get_watchtower_market_snapshot()
        self.assertIn("^GSPC", result["indices"])
        self.assertAlmostEqual(result["indices"]["^GSPC"]["last"], 5100.0)
        expected_pct = (5100.0 - 5000.0) / 5000.0 * 100.0
        self.assertAlmostEqual(result["indices"]["^GSPC"]["pct"], expected_pct, places=4)

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_prev_val_zero_pct_is_zero(self, mock_market_cls, mock_download):
        mock_market_cls.return_value = MagicMock(status={})

        close_series = pd.Series([0.0, 100.0], index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]))
        hist = pd.DataFrame({"Close": [0]})
        hist.get = lambda key, default=None: close_series if key == "Close" else default
        mock_download.return_value = hist

        result = get_watchtower_market_snapshot()
        self.assertEqual(result["indices"]["^GSPC"]["pct"], 0.0)

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_insufficient_data_points(self, mock_market_cls, mock_download):
        """Only 1 data point → no index entry."""
        mock_market_cls.return_value = MagicMock(status={})

        close_df = pd.DataFrame(
            {"^GSPC": [4800.0], "^IXIC": [15000.0], "^DJI": [37000.0], "^SET.BK": [1400.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )
        hist = pd.DataFrame({"Close": [0]})
        hist.get = lambda key, default=None: close_df if key == "Close" else default
        mock_download.return_value = hist

        result = get_watchtower_market_snapshot()
        self.assertEqual(result["indices"], {})

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_missing_symbol_in_close_frame(self, mock_market_cls, mock_download):
        """Close frame only has some symbols."""
        mock_market_cls.return_value = MagicMock(status={})

        close_df = pd.DataFrame(
            {"^GSPC": [4800.0, 4850.0]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
        )
        hist = pd.DataFrame({"Close": [0]})
        hist.get = lambda key, default=None: close_df if key == "Close" else default
        mock_download.return_value = hist

        result = get_watchtower_market_snapshot()
        self.assertIn("^GSPC", result["indices"])
        self.assertNotIn("^IXIC", result["indices"])

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_close_frame_is_none(self, mock_market_cls, mock_download):
        """hist.get('Close') returns None → no indices."""
        mock_market_cls.return_value = MagicMock(status={})
        hist = pd.DataFrame()
        mock_download.return_value = hist

        result = get_watchtower_market_snapshot()
        self.assertEqual(result["indices"], {})

    @patch("sniper.news.yf.download")
    @patch("sniper.news.yf.Market")
    def test_status_none_values(self, mock_market_cls, mock_download):
        mock_market = MagicMock()
        mock_market.status = {"status": None, "message": None}
        mock_market_cls.return_value = mock_market
        mock_download.return_value = pd.DataFrame()

        result = get_watchtower_market_snapshot()
        self.assertEqual(result["us_status"], "Unknown")
        self.assertEqual(result["us_message"], "")


class TestFetchNewsForTicker(unittest.TestCase):
    """Tests for fetch_news_for_ticker — Finnhub news with relevance filtering."""

    def _make_finnhub_article(self, headline, summary, source="Reuters",
                              url="https://reuters.com/article", ts=1704067200):
        return {
            "headline": headline,
            "summary": summary,
            "source": source,
            "url": url,
            "datetime": ts,
        }

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_returns_relevant_articles(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple Inc.", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {"longName": "Apple Inc.", "shortName": "Apple"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "AAPL stock surges on earnings beat",
                "Apple Inc. reported strong earnings, stock rose 5%.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "AAPL stock surges on earnings beat")

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_empty_ticker_returns_empty(self, mock_search, mock_ticker, mock_get):
        result = fetch_news_for_ticker("", "fake_key")
        self.assertEqual(result, [])
        mock_get.assert_not_called()

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_none_ticker_returns_empty(self, mock_search, mock_ticker, mock_get):
        result = fetch_news_for_ticker(None, "fake_key")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_no_api_key_returns_empty(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "VRT", "shortname": "Vertiv", "longname": "Vertiv Holdings", "exchange": "NYSE"}
        ]
        mock_ticker.return_value.info = {}
        result = fetch_news_for_ticker("VRT", "")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_filters_irrelevant_no_ticker_match(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple Inc.", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "Weather looks great today",
                "Sunny skies across the nation.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_filters_no_finance_keyword(self, mock_search, mock_ticker, mock_get):
        """Article mentions ticker but no finance keywords → filtered out."""
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple Inc.", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "AAPL celebrates new product launch",
                "Apple launches new headphones with great sound.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_company_name_match_passes_relevance(self, mock_search, mock_ticker, mock_get):
        """Article mentions company longname (not ticker) + finance keyword → relevant."""
        mock_search.return_value.quotes = [
            {"symbol": "VRT", "shortname": "Vertiv", "longname": "Vertiv Holdings", "exchange": "NYSE"}
        ]
        mock_ticker.return_value.info = {"longName": "Vertiv Holdings", "shortName": "Vertiv"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "Vertiv Holdings reports strong quarterly earnings",
                "Vertiv Holdings Q3 revenue beats analyst expectations.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("VRT", "fake_key")
        self.assertEqual(len(result), 1)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_company_alias_match(self, mock_search, mock_ticker, mock_get):
        """Article mentions shortname alias + finance keyword → relevant."""
        mock_search.return_value.quotes = [
            {"symbol": "TSM", "shortname": "TSMC", "longname": "Taiwan Semiconductor", "exchange": "NYSE"}
        ]
        mock_ticker.return_value.info = {"longName": "Taiwan Semiconductor", "shortName": "TSMC"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "TSMC revenue jumps on AI chip demand",
                "TSMC quarterly profit surges as investor sentiment improves.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("TSM", "fake_key")
        self.assertEqual(len(result), 1)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_dollar_sign_ticker_variant(self, mock_search, mock_ticker, mock_get):
        """Article contains $AAPL format → matches ticker."""
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "$AAPL trading hits new highs",
                "Investors react as $AAPL stock reaches new highs.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(len(result), 1)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_parenthesized_ticker_variant(self, mock_search, mock_ticker, mock_get):
        """Article contains (VRT) format → matches ticker."""
        mock_search.return_value.quotes = [
            {"symbol": "VRT", "shortname": "Vertiv", "longname": "Vertiv Holdings", "exchange": "NYSE"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "Vertiv (VRT) analyst upgrades stock rating",
                "Wall Street analyst upgrades (VRT) to market outperform.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("VRT", "fake_key")
        self.assertEqual(len(result), 1)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_max_8_articles_returned(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        articles = [
            self._make_finnhub_article(
                f"AAPL stock earnings report #{i}",
                f"Apple stock market analysis #{i}.",
            )
            for i in range(15)
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = articles
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertLessEqual(len(result), 8)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_non_200_tries_next_candidate(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "PTT.BK", "shortname": "PTT", "longname": "PTT PCL", "exchange": "SET"}
        ]
        mock_ticker.return_value.info = {}

        resp_fail = MagicMock()
        resp_fail.status_code = 429

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.content = b"data"
        resp_ok.json.return_value = [
            self._make_finnhub_article(
                "PTT stock market update on earnings",
                "PTT PCL investor guidance improves.",
            ),
        ]
        mock_get.side_effect = [resp_fail, resp_ok]

        result = fetch_news_for_ticker("PTT.BK", "fake_key")
        self.assertEqual(len(result), 1)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_empty_payload_tries_next_candidate(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "MU", "shortname": "Micron", "longname": "Micron Technology", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        resp_empty = MagicMock()
        resp_empty.status_code = 200
        resp_empty.content = b"data"
        resp_empty.json.return_value = []

        mock_get.return_value = resp_empty

        result = fetch_news_for_ticker("MU", "fake_key")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_request_exception_continues(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}
        mock_get.side_effect = Exception("connection timeout")

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_search_exception_still_returns_empty(self, mock_search, mock_ticker, mock_get):
        mock_search.side_effect = Exception("search broken")
        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_allowed_sources_filtering(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "AAPL stock trading surges", "Apple earnings beat expectations.",
                source="Bloomberg",
            ),
            self._make_finnhub_article(
                "AAPL investor update on dividends", "Apple dividend announcement.",
                source="RandomBlog",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key", allowed_sources={"bloomberg"})
        # Only Bloomberg article should pass
        self.assertEqual(len(result), 1)
        self.assertIn("surges", result[0]["title"])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_allowed_domains_filtering(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "AAPL stock earnings update", "Market analysts weigh in.",
                source="Reuters", url="https://www.reuters.com/article/aapl",
            ),
            self._make_finnhub_article(
                "AAPL investor market news", "Apple revenue guidance raised.",
                source="SomeOther", url="https://www.randomblog.net/aapl",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key", allowed_domains={"reuters.com"})
        self.assertEqual(len(result), 1)
        self.assertIn("reuters.com", result[0]["url"])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_subdomain_matches_allowed_domain(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "AAPL stock analyst rating upgrade",
                "Market analyst upgrades Apple.",
                source="Yahoo", url="https://finance.yahoo.com/aapl",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key", allowed_domains={"yahoo.com"})
        self.assertEqual(len(result), 1)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_bk_ticker_strips_for_search(self, mock_search, mock_ticker, mock_get):
        """search_ticker for PTT.BK should be PTT (stripped)."""
        mock_search.return_value.quotes = [
            {"symbol": "PTT.BK", "shortname": "PTT", "longname": "PTT PCL", "exchange": "SET"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "PTT stock market quarterly earnings",
                "PTT PCL reports profit for the quarter.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("PTT.BK", "fake_key")
        self.assertEqual(len(result), 1)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_missing_headline_filtered(self, mock_search, mock_ticker, mock_get):
        """Articles without a headline are excluded."""
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            {"headline": "", "summary": "AAPL stock earnings update", "source": "Reuters",
             "url": "https://reuters.com/1", "datetime": 1704067200},
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_missing_url_filtered(self, mock_search, mock_ticker, mock_get):
        """Articles without a url are excluded."""
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            {"headline": "AAPL stock earnings", "summary": "Apple revenue guidance",
             "source": "Reuters", "url": "", "datetime": 1704067200},
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_yf_ticker_exception_still_works(self, mock_search, mock_ticker, mock_get):
        """If yf.Ticker raises, the function should still try Finnhub."""
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.side_effect = Exception("ticker fetch failed")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "AAPL stock market earnings report",
                "Apple stock trading volumes increase.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(len(result), 1)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_non_list_payload_skipped(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = {"error": "rate limited"}
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(result, [])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_non_dict_items_in_payload_skipped(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            "not_a_dict",
            42,
            self._make_finnhub_article(
                "AAPL stock market analyst report",
                "Apple earnings exceed market guidance.",
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(len(result), 1)

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_article_timestamp_conversion(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            self._make_finnhub_article(
                "AAPL stock earnings beat",
                "Apple market analysis shows strong revenue.",
                ts=1704067200,
            ),
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(len(result), 1)
        self.assertIn("Z", result[0]["publishedAt"])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_article_missing_datetime(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_resp.json.return_value = [
            {"headline": "AAPL stock market earnings update",
             "summary": "Apple revenue analyst forecast.",
             "source": "Reuters", "url": "https://reuters.com/1"},
        ]
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["publishedAt"])

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_each_finance_keyword_triggers_relevance(self, mock_search, mock_ticker, mock_get):
        """Each finance keyword individually makes an article relevant."""
        mock_search.return_value.quotes = [
            {"symbol": "VRT", "shortname": "Vertiv", "longname": "Vertiv Holdings", "exchange": "NYSE"}
        ]
        mock_ticker.return_value.info = {}

        keywords = [
            "STOCK", "SHARE", "EARNINGS", "REVENUE", "PROFIT", "LOSS", "GUIDANCE",
            "MARKET", "INVESTOR", "TRADING", "DIVIDEND", "ANALYST", "QUARTER", "FINANC"
        ]
        for kw in keywords:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"data"
            mock_resp.json.return_value = [
                self._make_finnhub_article(
                    f"VRT update on {kw.lower()} news",
                    f"Vertiv Holdings sees {kw.lower()} improvement.",
                ),
            ]
            mock_get.return_value = mock_resp

            result = fetch_news_for_ticker("VRT", "fake_key")
            self.assertGreaterEqual(len(result), 1, f"Keyword '{kw}' should trigger relevance")

    @patch("sniper.news.requests.get")
    @patch("sniper.news.yf.Ticker")
    @patch("sniper.news.yf.Search")
    def test_no_content_in_response_returns_empty(self, mock_search, mock_ticker, mock_get):
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "shortname": "Apple", "longname": "Apple Inc.", "exchange": "NMS"}
        ]
        mock_ticker.return_value.info = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b""
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        result = fetch_news_for_ticker("AAPL", "fake_key")
        self.assertEqual(result, [])


class TestGetEarningsDates(unittest.TestCase):
    """Tests for get_earnings_dates — yfinance earnings extraction."""

    @patch("sniper.news.yf.Ticker")
    def test_single_ticker_with_earnings(self, mock_ticker_cls):
        mock_obj = MagicMock()
        mock_obj.info = {"earningsDate": "2024-07-25"}
        mock_ticker_cls.return_value = mock_obj

        result = get_earnings_dates(["AAPL"])
        self.assertEqual(result, {"AAPL": "2024-07-25"})

    @patch("sniper.news.yf.Ticker")
    def test_multiple_tickers(self, mock_ticker_cls):
        dates = {"AAPL": "2024-07-25", "MSFT": "2024-08-01"}

        def make_ticker(sym):
            m = MagicMock()
            m.info = {"earningsDate": dates[sym]}
            return m

        mock_ticker_cls.side_effect = lambda sym: make_ticker(sym)

        result = get_earnings_dates(["AAPL", "MSFT"])
        self.assertEqual(len(result), 2)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)

    @patch("sniper.news.yf.Ticker")
    def test_ticker_without_earnings_key(self, mock_ticker_cls):
        mock_obj = MagicMock()
        mock_obj.info = {"longName": "Apple Inc."}
        mock_ticker_cls.return_value = mock_obj

        result = get_earnings_dates(["AAPL"])
        self.assertEqual(result, {})

    @patch("sniper.news.yf.Ticker")
    def test_ticker_exception_skipped(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("API error")

        result = get_earnings_dates(["AAPL"])
        self.assertEqual(result, {})

    @patch("sniper.news.yf.Ticker")
    def test_empty_tickers_list(self, mock_ticker_cls):
        result = get_earnings_dates([])
        self.assertEqual(result, {})
        mock_ticker_cls.assert_not_called()

    @patch("sniper.news.yf.Ticker")
    def test_mixed_success_and_failure(self, mock_ticker_cls):
        def side_effect(sym):
            if sym == "FAIL":
                raise Exception("API error")
            m = MagicMock()
            m.info = {"earningsDate": "2024-08-01"}
            return m

        mock_ticker_cls.side_effect = side_effect

        result = get_earnings_dates(["AAPL", "FAIL", "MSFT"])
        self.assertEqual(len(result), 2)
        self.assertIn("AAPL", result)
        self.assertNotIn("FAIL", result)
        self.assertIn("MSFT", result)

    @patch("sniper.news.yf.Ticker")
    def test_ticker_no_info_attribute(self, mock_ticker_cls):
        mock_obj = MagicMock(spec=[])  # no attributes
        mock_ticker_cls.return_value = mock_obj

        result = get_earnings_dates(["AAPL"])
        self.assertEqual(result, {})

    @patch("sniper.news.yf.Ticker")
    def test_earnings_date_as_list(self, mock_ticker_cls):
        mock_obj = MagicMock()
        mock_obj.info = {"earningsDate": ["2024-07-25", "2024-07-26"]}
        mock_ticker_cls.return_value = mock_obj

        result = get_earnings_dates(["AAPL"])
        self.assertEqual(result["AAPL"], ["2024-07-25", "2024-07-26"])


if __name__ == "__main__":
    unittest.main()
