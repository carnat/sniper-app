"""Tests for sniper/csv_import.py — CSV import parsing and helpers."""

import io
import unittest

import pandas as pd

from sniper.csv_import import (
    normalize_column_name,
    get_first_matching_column,
    normalize_asset_class,
    looks_thai_market_hint,
    parse_numeric_value,
    parse_date_value,
    build_import_key,
    import_key_exists,
    parse_transactions_csv,
    get_csv_templates,
)


class TestNormalizeColumnName(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(normalize_column_name("Symbol"), "symbol")

    def test_spaces_and_underscores(self):
        self.assertEqual(normalize_column_name("Trade Date"), "tradedate")
        self.assertEqual(normalize_column_name("trade_date"), "tradedate")

    def test_leading_trailing_spaces(self):
        self.assertEqual(normalize_column_name("  Ticker  "), "ticker")


class TestGetFirstMatchingColumn(unittest.TestCase):
    def test_finds_match(self):
        df = pd.DataFrame({"Symbol": [1], "Price": [2]})
        result = get_first_matching_column(df, ["ticker", "symbol"])
        self.assertEqual(result, "Symbol")

    def test_returns_none_when_no_match(self):
        df = pd.DataFrame({"Name": [1]})
        result = get_first_matching_column(df, ["ticker", "symbol"])
        self.assertIsNone(result)

    def test_underscore_vs_space_fuzzy(self):
        df = pd.DataFrame({"Trade Date": ["2026-01-01"]})
        result = get_first_matching_column(df, ["trade_date"])
        self.assertEqual(result, "Trade Date")


class TestNormalizeAssetClass(unittest.TestCase):
    def test_us_stock_variants(self):
        for v in ["US Stock", "us", "stock", "equity", "us_equity"]:
            self.assertEqual(normalize_asset_class(v), "US Stock", f"Failed for {v}")

    def test_thai_stock_variants(self):
        for v in ["Thai Stock", "th stock", "thai", "thai_equity"]:
            self.assertEqual(normalize_asset_class(v), "Thai Stock", f"Failed for {v}")

    def test_mutual_fund_variants(self):
        for v in ["Mutual Fund", "fund", "thai fund", "th fund"]:
            self.assertEqual(normalize_asset_class(v), "Mutual Fund", f"Failed for {v}")

    def test_unknown_returns_none(self):
        self.assertIsNone(normalize_asset_class("crypto"))
        self.assertIsNone(normalize_asset_class(""))


class TestLooksThaiMarketHint(unittest.TestCase):
    def test_thb_currency(self):
        self.assertTrue(looks_thai_market_hint("THB", ""))

    def test_set_market(self):
        self.assertTrue(looks_thai_market_hint(None, "SET"))

    def test_bangkok_market(self):
        self.assertTrue(looks_thai_market_hint("", "Bangkok Stock Exchange"))

    def test_bk_market(self):
        self.assertTrue(looks_thai_market_hint("", "BK"))

    def test_us_market(self):
        self.assertFalse(looks_thai_market_hint("USD", "NASDAQ"))

    def test_none_values(self):
        self.assertFalse(looks_thai_market_hint(None, None))


class TestParseNumericValue(unittest.TestCase):
    def test_simple_number(self):
        self.assertEqual(parse_numeric_value("123.45"), 123.45)

    def test_comma_separated(self):
        self.assertEqual(parse_numeric_value("1,234.56"), 1234.56)

    def test_dollar_sign(self):
        self.assertEqual(parse_numeric_value("$100.00"), 100.0)

    def test_thai_baht(self):
        self.assertEqual(parse_numeric_value("฿200.50"), 200.50)

    def test_parentheses_negative(self):
        self.assertEqual(parse_numeric_value("(500.00)"), -500.0)

    def test_none_value(self):
        self.assertIsNone(parse_numeric_value(None))

    def test_empty_string(self):
        self.assertIsNone(parse_numeric_value(""))

    def test_nan_float(self):
        self.assertIsNone(parse_numeric_value(float("nan")))

    def test_currency_prefix_thb(self):
        self.assertEqual(parse_numeric_value("THB 100"), 100.0)

    def test_invalid_text(self):
        self.assertIsNone(parse_numeric_value("abc"))


class TestParseDateValue(unittest.TestCase):
    def test_standard_date(self):
        self.assertEqual(parse_date_value("2026-01-15"), "2026-01-15")

    def test_compact_yyyymmdd(self):
        self.assertEqual(parse_date_value("20260115"), "2026-01-15")

    def test_compact_with_dot_zero(self):
        self.assertEqual(parse_date_value("20260115.0"), "2026-01-15")

    def test_other_format(self):
        result = parse_date_value("Jan 15, 2026")
        self.assertEqual(result, "2026-01-15")

    def test_none_value(self):
        self.assertIsNone(parse_date_value(None))

    def test_empty_string(self):
        self.assertIsNone(parse_date_value(""))

    def test_invalid_date(self):
        self.assertIsNone(parse_date_value("not-a-date"))

    def test_nan_float(self):
        self.assertIsNone(parse_date_value(float("nan")))


class TestBuildImportKey(unittest.TestCase):
    def test_basic_key(self):
        key = build_import_key("US Stock", "Buy", "AAPL", "", 10.0, 190.5, "2026-01-15")
        self.assertEqual(key, "US STOCK|BUY|AAPL|10.00000000|190.50000000|2026-01-15")

    def test_fund_uses_fund_code(self):
        key = build_import_key("Mutual Fund", "Buy", "", "SCBNDQ(E)", 1000, 13.5, "2026-01-05")
        self.assertIn("SCBNDQ(E)", key)
        self.assertNotIn("||", key)

    def test_deterministic(self):
        k1 = build_import_key("US Stock", "Buy", "VRT", "", 5, 100, "2026-01-01")
        k2 = build_import_key("US Stock", "Buy", "VRT", "", 5, 100, "2026-01-01")
        self.assertEqual(k1, k2)


class TestImportKeyExists(unittest.TestCase):
    def test_found(self):
        history = [{"import_key": "KEY1"}, {"import_key": "KEY2"}]
        self.assertTrue(import_key_exists("KEY1", history))

    def test_not_found(self):
        history = [{"import_key": "KEY1"}]
        self.assertFalse(import_key_exists("KEY3", history))

    def test_empty_key(self):
        self.assertFalse(import_key_exists("", [{"import_key": "KEY1"}]))

    def test_empty_history(self):
        self.assertFalse(import_key_exists("KEY1", []))


class TestParseTransactionsCsv(unittest.TestCase):
    def test_parse_us_transactions(self):
        csv_text = "symbol,action,quantity,price,date\nAAPL,BUY,10,190.50,2026-01-15\nAAPL,SELL,2,201.20,2026-02-01\n"
        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(len(errors), 0)
        self.assertEqual(df.iloc[0]["action"], "Buy")
        self.assertEqual(df.iloc[1]["action"], "Sell")

    def test_parse_holdings_mode(self):
        csv_text = "symbol,shares,avg_cost\nAAPL,10,190.50\nMSFT,5,420.00\n"
        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["action"], "Buy")

    def test_thai_stock_detection(self):
        csv_text = "symbol,action,quantity,price,date\nADVANC.BK,BUY,50,220.00,2026-01-10\n"
        df, errors = parse_transactions_csv(io.StringIO(csv_text), "Auto-detect")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["asset_class"], "Thai Stock")

    def test_fund_code_column(self):
        csv_text = "fund_code,action,quantity,price,date,master\nSCBNDQ(E),BUY,1000,13.50,2026-01-05,QQQ\n"
        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["asset_class"], "Mutual Fund")
        self.assertEqual(df.iloc[0]["fund_code"], "SCBNDQ(E)")
        self.assertEqual(df.iloc[0]["master"], "QQQ")

    def test_missing_columns_errors(self):
        csv_text = "name,value\ntest,123\n"
        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")
        self.assertIsNone(df)
        self.assertTrue(len(errors) > 0)

    def test_cost_basis_fallback(self):
        csv_text = "symbol,shares,cost_basis\nAAPL,10,1905.00\n"
        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["price"], 190.5)

    def test_skips_cash_rows(self):
        csv_text = "symbol,action,quantity,price,date\n$$CASH,BUY,100,1.0,2026-01-01\nAAPL,BUY,10,190,2026-01-15\n"
        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "AAPL")

    def test_thai_stock_appends_bk_suffix(self):
        csv_text = "symbol,action,quantity,price,date,currency\nADVANC,BUY,100,220,2026-01-10,THB\n"
        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")
        self.assertIsNotNone(df)
        self.assertEqual(df.iloc[0]["symbol"], "ADVANC.BK")

    def test_import_key_generated(self):
        csv_text = "symbol,action,quantity,price,date\nMSFT,BUY,3,420,2026-01-07\n"
        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")
        self.assertIsNotNone(df)
        self.assertTrue(len(df.iloc[0]["import_key"]) > 0)


class TestGetCsvTemplates(unittest.TestCase):
    def test_returns_all_templates(self):
        templates = get_csv_templates()
        self.assertIn("us", templates)
        self.assertIn("thai_stock", templates)
        self.assertIn("thai_fund", templates)
        self.assertIn("mixed", templates)

    def test_templates_are_valid_csv(self):
        templates = get_csv_templates()
        for name, csv_text in templates.items():
            df = pd.read_csv(io.StringIO(csv_text))
            self.assertTrue(len(df) > 0, f"Template '{name}' is empty")


if __name__ == "__main__":
    unittest.main()
