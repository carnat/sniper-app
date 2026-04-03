"""Tests for sniper/alerts.py — price alert engine."""

import unittest
from datetime import datetime, timedelta

from sniper.alerts import check_price_alerts


class TestCheckPriceAlerts(unittest.TestCase):
    def _make_portfolio(self, tickers, avg_costs):
        return {"Ticker": tickers, "Avg_Cost": avg_costs}

    def test_no_alerts_below_threshold(self):
        portfolio = self._make_portfolio(["AAPL"], [100.0])
        alerts, state = check_price_alerts(portfolio, [103.0], "US Stock", {}, threshold=5)
        # 3% change is below 5% threshold — no alerts generated
        self.assertEqual(len(alerts), 0)

    def test_alert_triggered_above_threshold(self):
        portfolio = self._make_portfolio(["AAPL"], [100.0])
        alerts, state = check_price_alerts(portfolio, [106.0], "US Stock", {}, threshold=5)
        self.assertEqual(len(alerts), 1)
        self.assertTrue(alerts[0]["is_new"])
        self.assertAlmostEqual(alerts[0]["change_pct"], 6.0)
        self.assertEqual(alerts[0]["change_type"], "📈 UP")
        self.assertEqual(alerts[0]["currency_symbol"], "$")

    def test_alert_triggered_below_threshold_negative(self):
        portfolio = self._make_portfolio(["AAPL"], [100.0])
        alerts, state = check_price_alerts(portfolio, [93.0], "US Stock", {}, threshold=5)
        self.assertEqual(len(alerts), 1)
        self.assertTrue(alerts[0]["is_new"])
        self.assertAlmostEqual(alerts[0]["change_pct"], -7.0)
        self.assertEqual(alerts[0]["change_type"], "📉 DOWN")

    def test_thai_stock_currency_symbol(self):
        portfolio = self._make_portfolio(["ADVANC.BK"], [200.0])
        alerts, state = check_price_alerts(portfolio, [220.0], "Thai Stock", {}, threshold=5)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["currency_symbol"], "฿")

    def test_skips_zero_avg_cost(self):
        portfolio = self._make_portfolio(["AAPL", "MSFT"], [0, 300.0])
        alerts, state = check_price_alerts(portfolio, [150.0, 320.0], "US Stock", {}, threshold=5)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["ticker"], "MSFT")

    def test_skips_none_avg_cost(self):
        portfolio = self._make_portfolio(["AAPL"], [None])
        alerts, state = check_price_alerts(portfolio, [150.0], "US Stock", {}, threshold=5)
        self.assertEqual(len(alerts), 0)

    def test_cooldown_blocks_repeat_alert(self):
        portfolio = self._make_portfolio(["AAPL"], [100.0])
        # First alert
        alerts1, state = check_price_alerts(portfolio, [110.0], "US Stock", {}, threshold=5, cooldown_minutes=60)
        self.assertTrue(alerts1[0]["is_new"])

        # Second call within cooldown — same direction — is_new should be False
        alerts2, state2 = check_price_alerts(portfolio, [112.0], "US Stock", state, threshold=5, cooldown_minutes=60)
        self.assertEqual(len(alerts2), 1)
        self.assertFalse(alerts2[0]["is_new"])

    def test_direction_change_triggers_new_alert(self):
        portfolio = self._make_portfolio(["AAPL"], [100.0])
        # First: up alert
        _, state = check_price_alerts(portfolio, [110.0], "US Stock", {}, threshold=5)
        # Then: down alert (direction changed)
        now_iso = datetime.now().isoformat()
        # Force cooldown to pass by backdating
        state_key = "US Stock|AAPL"
        state[state_key]["last_alert_at"] = (datetime.now() - timedelta(hours=2)).isoformat()

        alerts2, _ = check_price_alerts(portfolio, [90.0], "US Stock", state, threshold=5, cooldown_minutes=60)
        self.assertEqual(len(alerts2), 1)
        self.assertTrue(alerts2[0]["is_new"])

    def test_state_updated_correctly(self):
        portfolio = self._make_portfolio(["AAPL"], [100.0])
        _, state = check_price_alerts(portfolio, [110.0], "US Stock", {}, threshold=5)
        key = "US Stock|AAPL"
        self.assertIn(key, state)
        self.assertTrue(state[key]["triggered"])
        self.assertEqual(state[key]["direction"], "up")
        self.assertAlmostEqual(state[key]["last_pct"], 10.0)

    def test_state_reset_when_below_threshold(self):
        portfolio = self._make_portfolio(["AAPL"], [100.0])
        _, state = check_price_alerts(portfolio, [110.0], "US Stock", {}, threshold=5)
        # Price returns to within threshold
        _, state2 = check_price_alerts(portfolio, [102.0], "US Stock", state, threshold=5)
        key = "US Stock|AAPL"
        self.assertFalse(state2[key]["triggered"])

    def test_multiple_tickers(self):
        portfolio = self._make_portfolio(["AAPL", "MSFT", "VRT"], [100, 200, 50])
        alerts, state = check_price_alerts(
            portfolio, [110.0, 202.0, 58.0], "US Stock", {}, threshold=5
        )
        # AAPL: 10% up -> alert, MSFT: 1% -> no alert, VRT: 16% -> alert
        self.assertEqual(len(alerts), 2)
        tickers = [a["ticker"] for a in alerts]
        self.assertIn("AAPL", tickers)
        self.assertIn("VRT", tickers)

    def test_empty_portfolio(self):
        portfolio = self._make_portfolio([], [])
        alerts, state = check_price_alerts(portfolio, [], "US Stock", {}, threshold=5)
        self.assertEqual(len(alerts), 0)
        self.assertEqual(state, {})


if __name__ == "__main__":
    unittest.main()
