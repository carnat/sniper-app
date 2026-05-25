"""Extended tests for sniper.csv_import parsing helpers."""

import io
import unittest

import pandas as pd

from sniper.csv_import import (
    build_import_key,
    get_csv_templates,
    import_key_exists,
    parse_transactions_csv,
)


class TestParseTransactionsCsvExtended(unittest.TestCase):
    def test_transaction_mode_parses_buy_and_sell_rows(self):
        csv_text = (
            "symbol,action,quantity,price,date\n"
            "AAPL,BUY,10,190.5,2026-01-15\n"
            "AAPL,SELL,2,201.2,2026-02-01\n"
        )

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(errors, [])
        self.assertEqual(df["action"].tolist(), ["Buy", "Sell"])
        self.assertEqual(df["symbol"].tolist(), ["AAPL", "AAPL"])

    def test_holdings_mode_infers_buy_from_shares_and_avg_cost(self):
        csv_text = "symbol,shares,avg_cost\nAAPL,10,190.5\nMSFT,5,420\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(errors, [])
        self.assertEqual(df["action"].tolist(), ["Buy", "Buy"])
        self.assertEqual(df["price"].tolist(), [190.5, 420.0])

    def test_returns_error_when_quantity_column_missing(self):
        csv_text = "symbol,action,price\nAAPL,BUY,190.5\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertIsNone(df)
        self.assertTrue(any("Missing quantity column" in error for error in errors))

    def test_returns_error_when_price_and_cost_basis_missing(self):
        csv_text = "symbol,action,quantity\nAAPL,BUY,10\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertIsNone(df)
        self.assertTrue(any("Missing price or cost basis column" in error for error in errors))

    def test_returns_error_when_symbol_and_fund_code_missing(self):
        csv_text = "action,quantity,price\nBUY,10,190.5\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertIsNone(df)
        self.assertTrue(any("Missing instrument column" in error for error in errors))

    def test_invalid_action_generates_row_error_and_skips_row(self):
        csv_text = (
            "symbol,action,quantity,price,date\n"
            "AAPL,HOLD,10,190.5,2026-01-15\n"
            "MSFT,BUY,5,420,2026-01-16\n"
        )

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "MSFT")
        self.assertTrue(any("Unsupported action 'HOLD'" in error for error in errors))

    def test_zero_or_negative_quantity_with_inferred_action_is_silently_skipped(self):
        csv_text = "symbol,action,quantity,price\nAAPL,,0,190.5\nMSFT,, -5,420\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertIsNotNone(df)
        self.assertTrue(df.empty)
        self.assertEqual(errors, [])

    def test_zero_quantity_with_explicit_action_generates_row_error(self):
        csv_text = "symbol,action,quantity,price\nAAPL,BUY,0,190.5\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertTrue(df.empty)
        self.assertEqual(errors, ["Row 2: Invalid quantity"])

    def test_cost_basis_fallback_computes_price(self):
        csv_text = "symbol,shares,cost_basis\nAAPL,10,1905\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(errors, [])
        self.assertAlmostEqual(df.iloc[0]["price"], 190.5)

    def test_thai_stock_auto_detects_from_thb_currency(self):
        csv_text = "symbol,action,quantity,price,currency\nADVANC,BUY,100,220,THB\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(errors, [])
        self.assertEqual(df.iloc[0]["asset_class"], "Thai Stock")

    def test_thai_stock_appends_bk_suffix(self):
        csv_text = "symbol,action,quantity,price,currency\nADVANC,BUY,100,220,THB\n"

        df, _ = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(df.iloc[0]["symbol"], "ADVANC.BK")

    def test_fund_code_column_forces_mutual_fund_asset_class(self):
        csv_text = "fund_code,action,quantity,price,date\nSCBNDQ(E),BUY,1000,13.5,2026-01-05\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(errors, [])
        self.assertEqual(df.iloc[0]["asset_class"], "Mutual Fund")
        self.assertEqual(df.iloc[0]["fund_code"], "SCBNDQ(E)")

    def test_master_column_is_populated(self):
        csv_text = "fund_code,action,quantity,price,date,master\nSCBNDQ(E),BUY,1000,13.5,2026-01-05,qqq\n"

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(errors, [])
        self.assertEqual(df.iloc[0]["master"], "QQQ")

    def test_date_parsing_accepts_valid_dates_and_rejects_invalid_dates(self):
        csv_text = (
            "symbol,action,quantity,price,date\n"
            "AAPL,BUY,10,190.5,20260115\n"
            "MSFT,BUY,5,420,not-a-date\n"
        )

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["transaction_date"], "2026-01-15")
        self.assertTrue(any("Invalid date 'not-a-date'" in error for error in errors))

    def test_cash_and_cash_transaction_rows_are_skipped(self):
        csv_text = (
            "symbol,transaction_type,quantity,price,date\n"
            "$$CASH,BUY,100,1.0,2026-01-01\n"
            "USD,DEPOSIT,1,1.0,2026-01-02\n"
            "AAPL,BUY,10,190.5,2026-01-15\n"
        )

        df, errors = parse_transactions_csv(io.StringIO(csv_text), "US Stock")

        self.assertEqual(errors, [])
        self.assertEqual(df["symbol"].tolist(), ["AAPL"])


class TestCsvImportHelpersExtended(unittest.TestCase):
    def test_build_import_key_is_deterministic(self):
        left = build_import_key("US Stock", "Buy", "AAPL", "", 10, 190.5, "2026-01-15")
        right = build_import_key("US Stock", "Buy", "AAPL", "", 10, 190.5, "2026-01-15")

        self.assertEqual(left, right)
        self.assertEqual(left, "US STOCK|BUY|AAPL|10.00000000|190.50000000|2026-01-15")

    def test_import_key_exists_finds_and_misses_keys(self):
        history = [{"import_key": "KEY-1"}, {"import_key": "KEY-2"}]

        self.assertTrue(import_key_exists("KEY-1", history))
        self.assertFalse(import_key_exists("KEY-3", history))

    def test_get_csv_templates_returns_expected_valid_csv_strings(self):
        templates = get_csv_templates()

        self.assertEqual(set(templates), {"us", "thai_stock", "thai_fund", "mixed"})
        for name, csv_text in templates.items():
            self.assertIsInstance(csv_text, str)
            parsed = pd.read_csv(io.StringIO(csv_text))
            self.assertFalse(parsed.empty, name)


if __name__ == "__main__":
    unittest.main()
