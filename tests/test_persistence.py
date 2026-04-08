"""Tests for sniper/persistence.py — file persistence layer."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sniper.persistence import (
    get_data_dir,
    get_transactions_file_path,
    get_alert_state_file_path,
    get_analytics_snapshots_file_path,
    get_scenario_library_file_path,
    get_watchlists_file_path,
    get_options_iv_history_file_path,
    get_calendar_events_file_path,
    load_transaction_history,
    save_transaction_history,
    load_alert_state,
    save_alert_state,
    load_analytics_snapshots,
    save_analytics_snapshots,
    load_saved_scenarios,
    save_saved_scenarios,
    load_watchlists,
    save_watchlists,
    load_options_iv_history,
    save_options_iv_history,
    load_calendar_events,
    save_calendar_events,
)


class PersistenceTestBase(unittest.TestCase):
    """Base class that patches data dir to temp directory."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.patcher = patch("sniper.persistence.get_data_dir", return_value=Path(self.tmp_dir))
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()


class TestTransactionHistory(PersistenceTestBase):
    def test_load_empty(self):
        result = load_transaction_history()
        self.assertEqual(result, [])

    def test_save_and_load(self):
        history = [
            {"date": "2026-01-15", "ticker": "AAPL", "type": "Buy", "shares": 10},
            {"date": "2026-02-01", "ticker": "AAPL", "type": "Sell", "shares": 2},
        ]
        save_transaction_history(history)
        loaded = load_transaction_history()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["ticker"], "AAPL")

    def test_load_corrupted_file(self):
        path = Path(self.tmp_dir) / "transactions.json"
        path.write_text("not valid json", encoding="utf-8")
        result = load_transaction_history()
        self.assertEqual(result, [])

    def test_load_non_list_data(self):
        path = Path(self.tmp_dir) / "transactions.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        result = load_transaction_history()
        self.assertEqual(result, [])


class TestAlertState(PersistenceTestBase):
    def test_load_empty(self):
        result = load_alert_state()
        self.assertEqual(result, {})

    def test_save_and_load(self):
        state = {"US Stock|AAPL": {"triggered": True, "direction": "up"}}
        save_alert_state(state)
        loaded = load_alert_state()
        self.assertEqual(loaded["US Stock|AAPL"]["triggered"], True)

    def test_load_corrupted_file(self):
        path = Path(self.tmp_dir) / "alert_state.json"
        path.write_text("{invalid", encoding="utf-8")
        result = load_alert_state()
        self.assertEqual(result, {})

    def test_load_non_dict_data(self):
        path = Path(self.tmp_dir) / "alert_state.json"
        path.write_text('[1, 2, 3]', encoding="utf-8")
        result = load_alert_state()
        self.assertEqual(result, {})


class TestAnalyticsSnapshots(PersistenceTestBase):
    def test_load_empty(self):
        self.assertEqual(load_analytics_snapshots(), [])

    def test_save_and_load(self):
        snapshots = [{"date": "2026-01-15", "total_value": 50000}]
        save_analytics_snapshots(snapshots)
        loaded = load_analytics_snapshots()
        self.assertEqual(len(loaded), 1)


class TestSavedScenarios(PersistenceTestBase):
    def test_load_empty(self):
        self.assertEqual(load_saved_scenarios(), [])

    def test_save_and_load(self):
        scenarios = [{"name": "Base Case", "params": {}}]
        save_saved_scenarios(scenarios)
        loaded = load_saved_scenarios()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["name"], "Base Case")


class TestWatchlists(PersistenceTestBase):
    def test_load_empty(self):
        self.assertEqual(load_watchlists(), {})

    def test_save_and_load(self):
        watchlists = {"Arsenal": ["VRT", "ASTS", "VST"]}
        save_watchlists(watchlists)
        loaded = load_watchlists()
        self.assertIn("Arsenal", loaded)
        self.assertEqual(loaded["Arsenal"], ["VRT", "ASTS", "VST"])

    def test_normalizes_symbols_to_upper(self):
        watchlists = {"MyList": ["aapl", " msft ", "vrt"]}
        save_watchlists(watchlists)
        loaded = load_watchlists()
        self.assertEqual(loaded["MyList"], ["AAPL", "MSFT", "VRT"])

    def test_filters_invalid_entries(self):
        path = Path(self.tmp_dir) / "watchlists.json"
        data = {"valid": ["AAPL"], 123: ["invalid_key"], "also_valid": "not_a_list"}
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_watchlists()
        self.assertIn("valid", loaded)
        self.assertNotIn(123, loaded)
        self.assertNotIn("also_valid", loaded)


class TestOptionsIvHistory(PersistenceTestBase):
    def test_load_empty(self):
        self.assertEqual(load_options_iv_history(), {})

    def test_save_and_load(self):
        history = {
            "AAPL": [
                {"date": "2026-01-01", "atm_iv": 0.20},
                {"date": "2026-01-02", "atm_iv": 0.22},
            ]
        }
        save_options_iv_history(history)
        loaded = load_options_iv_history()
        self.assertIn("AAPL", loaded)
        self.assertEqual(len(loaded["AAPL"]), 2)

    def test_filters_invalid_iv_values(self):
        path = Path(self.tmp_dir) / "options_iv_history.json"
        data = {
            "AAPL": [
                {"date": "2026-01-01", "atm_iv": 0.20},
                {"date": "2026-01-02", "atm_iv": "not_a_number"},
                {"date": "2026-01-03"},  # missing atm_iv
            ]
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_options_iv_history()
        self.assertEqual(len(loaded["AAPL"]), 1)

    def test_caps_at_400_entries(self):
        entries = [{"date": f"2024-01-{(i % 28) + 1:02d}", "atm_iv": 0.2 + i * 0.001} for i in range(500)]
        history = {"AAPL": entries}
        save_options_iv_history(history)
        loaded = load_options_iv_history()
        self.assertLessEqual(len(loaded["AAPL"]), 400)


class TestCalendarEvents(PersistenceTestBase):
    def test_load_empty(self):
        self.assertEqual(load_calendar_events(), [])

    def test_save_and_load(self):
        events = [
            {"id": "1", "date": "2026-04-15", "event_type": "Earnings",
             "title": "AAPL Earnings", "instrument": "AAPL", "details": "", "status": "Planned"}
        ]
        save_calendar_events(events)
        loaded = load_calendar_events()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["title"], "AAPL Earnings")

    def test_filters_events_without_date(self):
        path = Path(self.tmp_dir) / "calendar_events.json"
        data = [
            {"date": "2026-04-15", "title": "Valid"},
            {"date": "", "title": "Missing date"},
            {"title": "No date field"},
        ]
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_calendar_events()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["title"], "Valid")

    def test_sets_defaults_for_missing_fields(self):
        path = Path(self.tmp_dir) / "calendar_events.json"
        data = [{"date": "2026-04-15"}]
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_calendar_events()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["event_type"], "Reminder")
        self.assertEqual(loaded[0]["status"], "Planned")


class TestPathGetters(unittest.TestCase):
    """Tests for all path-getter helper functions."""

    def test_get_data_dir(self):
        result = get_data_dir()
        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path(".streamlit"))

    def test_get_transactions_file_path(self):
        result = get_transactions_file_path()
        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path(".streamlit") / "transactions.json")

    def test_get_alert_state_file_path(self):
        result = get_alert_state_file_path()
        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path(".streamlit") / "alert_state.json")

    def test_get_analytics_snapshots_file_path(self):
        result = get_analytics_snapshots_file_path()
        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path(".streamlit") / "analytics_snapshots.json")

    def test_get_scenario_library_file_path(self):
        result = get_scenario_library_file_path()
        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path(".streamlit") / "scenario_library.json")

    def test_get_watchlists_file_path(self):
        result = get_watchlists_file_path()
        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path(".streamlit") / "watchlists.json")

    def test_get_options_iv_history_file_path(self):
        result = get_options_iv_history_file_path()
        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path(".streamlit") / "options_iv_history.json")

    def test_get_calendar_events_file_path(self):
        result = get_calendar_events_file_path()
        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path(".streamlit") / "calendar_events.json")

    def test_all_paths_share_parent(self):
        paths = [
            get_transactions_file_path(),
            get_alert_state_file_path(),
            get_analytics_snapshots_file_path(),
            get_scenario_library_file_path(),
            get_watchlists_file_path(),
            get_options_iv_history_file_path(),
            get_calendar_events_file_path(),
        ]
        for p in paths:
            self.assertEqual(p.parent, get_data_dir())


if __name__ == "__main__":
    unittest.main()
