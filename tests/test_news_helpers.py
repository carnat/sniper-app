"""Tests for sniper/news.py — pure helper functions (no API calls)."""

import os
import unittest
from unittest.mock import patch

from sniper.news import (
    get_finnhub_api_key,
    _extract_domain,
    format_news_timestamp,
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


if __name__ == "__main__":
    unittest.main()
