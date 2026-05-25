"""Extended tests for sniper.persistence."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sniper.persistence import (
    load_analytics_snapshots,
    load_calendar_events,
    load_options_iv_history,
    load_saved_scenarios,
    load_watchlists,
    save_analytics_snapshots,
    save_saved_scenarios,
)


class PersistenceExtendedTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmp_dir)
        self.patcher = patch("sniper.persistence.get_data_dir", return_value=self.data_dir)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class TestAnalyticsSnapshotsExtended(PersistenceExtendedTestBase):
    def test_load_analytics_snapshots_missing_and_valid_file(self):
        self.assertEqual(load_analytics_snapshots(), [])

        path = self.data_dir / "analytics_snapshots.json"
        payload = [{"date": "2026-01-15", "total_value": 50000}]
        path.write_text(json.dumps(payload), encoding="utf-8")

        self.assertEqual(load_analytics_snapshots(), payload)

    def test_save_analytics_snapshots_roundtrip(self):
        snapshots = [{"date": "2026-01-15", "total_value": 50000}]

        save_analytics_snapshots(snapshots)

        self.assertEqual(load_analytics_snapshots(), snapshots)


class TestSavedScenariosExtended(PersistenceExtendedTestBase):
    def test_load_saved_scenarios_missing_and_valid_file(self):
        self.assertEqual(load_saved_scenarios(), [])

        path = self.data_dir / "scenario_library.json"
        payload = [{"name": "Stress", "params": {"drop": 0.25}}]
        path.write_text(json.dumps(payload), encoding="utf-8")

        self.assertEqual(load_saved_scenarios(), payload)

    def test_save_saved_scenarios_roundtrip(self):
        scenarios = [{"name": "Base Case", "params": {"dca": True}}]

        save_saved_scenarios(scenarios)

        self.assertEqual(load_saved_scenarios(), scenarios)


class TestCalendarEventsExtended(PersistenceExtendedTestBase):
    def test_load_calendar_events_cleans_fields_adds_uuid_defaults_status_and_caps_at_1200(self):
        path = self.data_dir / "calendar_events.json"
        events = [
            {"id": "drop-me", "date": "2026-01-01", "title": "Too Old"},
            *[
                {"id": f"event-{i}", "date": f"2026-02-{(i % 28) + 1:02d}", "title": f"Event {i}", "status": "Done"}
                for i in range(1, 1200)
            ],
            {"date": "2026-03-15", "event_type": "Review", "title": " rebalance ", "instrument": " aapl ", "details": "  note  "},
        ]
        path.write_text(json.dumps(events), encoding="utf-8")

        with patch("sniper.persistence.uuid.uuid4", return_value="generated-uuid"):
            loaded = load_calendar_events()

        self.assertEqual(len(loaded), 1200)
        self.assertEqual(loaded[-1]["id"], "generated-uuid")
        self.assertEqual(loaded[-1]["status"], "Planned")
        self.assertEqual(loaded[-1]["title"], "rebalance")
        self.assertEqual(loaded[-1]["instrument"], "AAPL")
        self.assertEqual(loaded[-1]["details"], "note")
        self.assertNotEqual(loaded[0]["id"], "drop-me")

    def test_load_calendar_events_skips_entries_without_date(self):
        path = self.data_dir / "calendar_events.json"
        path.write_text(
            json.dumps([
                {"date": "2026-03-15", "title": "Keep"},
                {"date": "", "title": "Skip"},
                {"title": "Skip too"},
            ]),
            encoding="utf-8",
        )

        loaded = load_calendar_events()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["title"], "Keep")

    def test_load_calendar_events_skips_non_dict_entries(self):
        path = self.data_dir / "calendar_events.json"
        path.write_text(json.dumps([{"date": "2026-03-15", "title": "Keep"}, "bad", 123, []]), encoding="utf-8")

        loaded = load_calendar_events()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["title"], "Keep")


class TestOptionsIvHistoryExtended(PersistenceExtendedTestBase):
    def test_load_options_iv_history_skips_invalid_floats_and_enforces_400_day_window(self):
        path = self.data_dir / "options_iv_history.json"
        values = [{"date": f"2025-01-{(i % 28) + 1:02d}", "atm_iv": i / 1000.0} for i in range(402)]
        values.extend([{"date": "2026-01-01", "atm_iv": "bad"}, {"date": "2026-01-02"}, "bad"])
        path.write_text(json.dumps({"AAPL": values}), encoding="utf-8")

        loaded = load_options_iv_history()

        self.assertEqual(len(loaded["AAPL"]), 400)
        self.assertEqual(loaded["AAPL"][0]["date"], values[2]["date"])
        self.assertTrue(all(isinstance(row["atm_iv"], float) for row in loaded["AAPL"]))

    def test_load_options_iv_history_skips_non_string_symbols(self):
        path = self.data_dir / "options_iv_history.json"
        path.write_text("{}", encoding="utf-8")

        with patch(
            "sniper.persistence.json.load",
            return_value={123: [{"date": "2026-01-01", "atm_iv": 0.2}], "AAPL": [{"date": "2026-01-01", "atm_iv": 0.3}], "MSFT": "bad"},
        ):
            loaded = load_options_iv_history()

        self.assertEqual(loaded, {"AAPL": [{"date": "2026-01-01", "atm_iv": 0.3}]})


class TestWatchlistsExtended(PersistenceExtendedTestBase):
    def test_load_watchlists_uppercases_symbols_removes_empty_values_and_skips_non_lists(self):
        path = self.data_dir / "watchlists.json"
        path.write_text("{}", encoding="utf-8")

        with patch(
            "sniper.persistence.json.load",
            return_value={123: ["aapl"], "Focus": [" aapl ", "", " msft "], "Invalid": "not-a-list"},
        ):
            loaded = load_watchlists()

        self.assertEqual(loaded, {"Focus": ["AAPL", "MSFT"]})


if __name__ == "__main__":
    unittest.main()
