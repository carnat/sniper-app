"""Tests for scripts/fetch_prices.py — regime and zone helpers."""

import sys
import unittest
from pathlib import Path

# Add parent directory for script imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_prices import (
    get_regime,
    get_thb_zone,
    _parse_news_item,
    _get_series,
    compute_gate,
    build_adv_section,
    build_output,
    WATCHTOWER_CORE,
    WATCHTOWER_SAT,
)


class TestGetRegime(unittest.TestCase):
    def test_green_low_vix(self):
        self.assertEqual(get_regime(15.0), "GREEN")

    def test_green_boundary(self):
        self.assertEqual(get_regime(21.99), "GREEN")

    def test_yellow_at_22(self):
        self.assertEqual(get_regime(22.0), "YELLOW")

    def test_yellow_at_24(self):
        self.assertEqual(get_regime(24.5), "YELLOW")

    def test_yellow_boundary(self):
        self.assertEqual(get_regime(24.99), "YELLOW")

    def test_orange_at_25(self):
        self.assertEqual(get_regime(25.0), "ORANGE")

    def test_orange_at_30(self):
        """VIX=30.0 is ORANGE (inclusive upper bound)."""
        self.assertEqual(get_regime(30.0), "ORANGE")

    def test_red_above_30(self):
        self.assertEqual(get_regime(30.01), "RED")

    def test_red_high_vix(self):
        self.assertEqual(get_regime(50.0), "RED")

    def test_zero_vix(self):
        self.assertEqual(get_regime(0.0), "GREEN")


class TestGetThbZone(unittest.TestCase):
    def test_zone_a_strong_baht(self):
        self.assertEqual(get_thb_zone(30.0), "A")

    def test_zone_a_boundary(self):
        self.assertEqual(get_thb_zone(31.99), "A")

    def test_zone_b_at_32(self):
        self.assertEqual(get_thb_zone(32.0), "B")

    def test_zone_b_at_36(self):
        self.assertEqual(get_thb_zone(36.0), "B")

    def test_zone_c_above_36(self):
        self.assertEqual(get_thb_zone(36.01), "C")

    def test_zone_c_weak_baht(self):
        self.assertEqual(get_thb_zone(40.0), "C")


class TestParseNewsItem(unittest.TestCase):
    def test_basic_legacy_format(self):
        item = {'title': 'Stock rises 5%', 'link': 'https://example.com', 'publisher': 'Reuters',
                'providerPublishTime': 1700000000}
        result = _parse_news_item(item, 'AAPL')
        self.assertIsNotNone(result)
        self.assertEqual(result['title'], 'Stock rises 5%')
        self.assertEqual(result['headline'], 'Stock rises 5%')
        self.assertEqual(result['url'], 'https://example.com')
        self.assertEqual(result['publisher'], 'Reuters')
        self.assertRegex(result['date'], r'^\d{4}-\d{2}-\d{2}$')

    def test_content_format(self):
        item = {'content': {
            'title': 'Earnings beat',
            'canonicalUrl': {'url': 'https://news.com/article'},
            'provider': {'displayName': 'Bloomberg'},
            'pubDate': '2024-06-15T10:00:00Z',
        }}
        result = _parse_news_item(item, 'MSFT')
        self.assertEqual(result['title'], 'Earnings beat')
        self.assertEqual(result['url'], 'https://news.com/article')
        self.assertEqual(result['publisher'], 'Bloomberg')

    def test_returns_none_for_empty_title(self):
        result = _parse_news_item({'title': ''}, 'AAPL')
        self.assertIsNone(result)
        result2 = _parse_news_item({}, 'AAPL')
        self.assertIsNone(result2)

    def test_title_truncated_at_140(self):
        item = {'title': 'A' * 200, 'link': '', 'publisher': ''}
        result = _parse_news_item(item, 'AAPL')
        self.assertEqual(len(result['title']), 140)

    def test_fallback_click_through_url(self):
        item = {'content': {
            'title': 'News',
            'clickThroughUrl': {'url': 'https://click.com'},
        }}
        result = _parse_news_item(item, 'X')
        self.assertEqual(result['url'], 'https://click.com')

    def test_no_timestamp_uses_pub_date(self):
        item = {'content': {'title': 'News', 'pubDate': '2024-03-20T12:00:00'}}
        result = _parse_news_item(item, 'AAPL')
        self.assertEqual(result['date'], '2024-03-20')


class TestGetSeries(unittest.TestCase):
    def test_single_index_df(self):
        import pandas as pd
        df = pd.DataFrame({'Close': [100, 101, 102]})
        series = _get_series(df, 'AAPL', 'Close')
        self.assertIsNotNone(series)
        self.assertEqual(len(series), 3)

    def test_single_index_missing_column(self):
        import pandas as pd
        df = pd.DataFrame({'Close': [100]})
        result = _get_series(df, 'AAPL', 'Volume')
        self.assertIsNone(result)

    def test_multi_index_df(self):
        import pandas as pd
        arrays = [['Close', 'Close', 'Volume', 'Volume'], ['AAPL', 'MSFT', 'AAPL', 'MSFT']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[150, 300, 1000, 2000]], columns=index)
        series = _get_series(df, 'AAPL', 'Close')
        self.assertIsNotNone(series)
        self.assertEqual(series.iloc[0], 150)

    def test_multi_index_missing_ticker(self):
        import pandas as pd
        arrays = [['Close', 'Close'], ['AAPL', 'MSFT']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[150, 300]], columns=index)
        result = _get_series(df, 'GOOG', 'Close')
        self.assertIsNone(result)

    def test_returns_none_on_empty_df(self):
        import pandas as pd
        df = pd.DataFrame()
        result = _get_series(df, 'AAPL', 'Close')
        self.assertIsNone(result)


class TestComputeGate(unittest.TestCase):
    def test_vix_freeze_true(self):
        prices = {'^VIX': {'price': 30.0}, 'THB=X': {'price': 34.0}}
        gate = compute_gate(prices)
        self.assertTrue(gate['vix_freeze'])
        self.assertEqual(gate['vix'], 30.0)

    def test_vix_freeze_false(self):
        prices = {'^VIX': {'price': 20.0}, 'THB=X': {'price': 34.0}}
        gate = compute_gate(prices)
        self.assertFalse(gate['vix_freeze'])

    def test_vix_freeze_boundary(self):
        prices = {'^VIX': {'price': 25.0}, 'THB=X': {'price': 34.0}}
        gate = compute_gate(prices)
        self.assertTrue(gate['vix_freeze'])

    def test_thb_zone_with_valid_rate(self):
        prices = {'^VIX': {'price': 15.0}, 'THB=X': {'price': 34.0}}
        gate = compute_gate(prices)
        self.assertEqual(gate['thb_zone'], 'B')
        self.assertEqual(gate['thb'], 34.0)

    def test_thb_zone_defaults_when_zero(self):
        prices = {'^VIX': {'price': 15.0}, 'THB=X': {'price': 0.0}}
        gate = compute_gate(prices)
        self.assertEqual(gate['thb_zone'], 'B')

    def test_missing_tickers(self):
        gate = compute_gate({})
        self.assertFalse(gate['vix_freeze'])
        self.assertEqual(gate['vix'], 0.0)
        self.assertEqual(gate['thb'], 0.0)


class TestBuildAdvSection(unittest.TestCase):
    def test_includes_watchtower_tickers(self):
        prices = {}
        for sym in WATCHTOWER_CORE + WATCHTOWER_SAT + ['FN']:
            prices[sym] = {'adv': 100000, 'volume': 200000}
        adv = build_adv_section(prices)
        self.assertGreater(len(adv), 0)
        for sym in adv:
            self.assertIn(sym, set(WATCHTOWER_CORE + WATCHTOWER_SAT + ['FN']))

    def test_gate_met_when_ratio_high(self):
        prices = {'TSEM': {'adv': 100000, 'volume': 200000}}
        adv = build_adv_section(prices)
        if 'TSEM' in adv:
            self.assertEqual(adv['TSEM']['adv_ratio'], 2.0)
            self.assertTrue(adv['TSEM']['gate_met'])

    def test_gate_not_met_when_ratio_low(self):
        prices = {'TSEM': {'adv': 100000, 'volume': 100000}}
        adv = build_adv_section(prices)
        if 'TSEM' in adv:
            self.assertEqual(adv['TSEM']['adv_ratio'], 1.0)
            self.assertFalse(adv['TSEM']['gate_met'])

    def test_skips_tickers_without_adv(self):
        prices = {'TSEM': {'volume': 100000}}
        adv = build_adv_section(prices)
        self.assertNotIn('TSEM', adv)

    def test_skips_tickers_without_volume(self):
        prices = {'TSEM': {'adv': 100000}}
        adv = build_adv_section(prices)
        self.assertNotIn('TSEM', adv)

    def test_zero_adv_gives_none_ratio(self):
        prices = {'TSEM': {'adv': 0, 'volume': 50000}}
        adv = build_adv_section(prices)
        if 'TSEM' in adv:
            self.assertIsNone(adv['TSEM']['adv_ratio'])
            self.assertFalse(adv['TSEM']['gate_met'])

    def test_empty_prices(self):
        adv = build_adv_section({})
        self.assertEqual(adv, {})

    def test_ignores_non_watchtower_tickers(self):
        prices = {'AAPL': {'adv': 5000000, 'volume': 10000000}}
        adv = build_adv_section(prices)
        self.assertNotIn('AAPL', adv)


if __name__ == "__main__":
    unittest.main()
