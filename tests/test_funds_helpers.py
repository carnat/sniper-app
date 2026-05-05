"""Tests for sniper/funds.py — fund data helpers (pure functions only)."""

import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd

from sniper.funds import (
    normalize_fund_token,
    resolve_fund_proj_id_and_class,
    get_sec_api_keys,
    _extract_nav,
    FUND_API_MAPPING,
    call_sec_api,
    load_disk_fund_registry,
    save_disk_fund_registry,
    build_fund_registry,
    fetch_fund_nav_with_previous,
    fetch_fund_nav,
    fetch_master_trends,
    get_master_data,
    get_fund_nav_by_code,
    get_fund_data,
)


class TestNormalizeFundToken(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(normalize_fund_token("SCBNDQ(E)"), "SCBNDQE")

    def test_normalizes_fund_code(self):
        self.assertEqual(normalize_fund_token("SCB-S&P500 FUND(SSF)"), "SCBS&P500FUNDSSF")

    def test_none_input(self):
        self.assertEqual(normalize_fund_token(None), "")

    def test_empty_string(self):
        self.assertEqual(normalize_fund_token(""), "")

    def test_case_insensitive(self):
        self.assertEqual(normalize_fund_token("scbndq"), "SCBNDQ")

    def test_dots_removed(self):
        self.assertEqual(normalize_fund_token("Fund.Name"), "FUNDNAME")


class TestResolveFundProjIdAndClass(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "SCBNDQ": "PROJ_001",
            "SCBS&P500FUND": "PROJ_002",
            "SCB70SSF": "PROJ_003",
        }

    def test_direct_match(self):
        proj_id, class_suffix = resolve_fund_proj_id_and_class("SCBNDQ", self.registry)
        self.assertEqual(proj_id, "PROJ_001")
        self.assertIsNone(class_suffix)

    def test_with_class_suffix(self):
        proj_id, class_suffix = resolve_fund_proj_id_and_class("SCBNDQ(E)", self.registry)
        self.assertEqual(proj_id, "PROJ_001")
        self.assertEqual(class_suffix, "(E)")

    def test_api_mapping_applied(self):
        proj_id, class_suffix = resolve_fund_proj_id_and_class("SCBS&P500(E)", self.registry)
        self.assertEqual(proj_id, "PROJ_002")
        self.assertEqual(class_suffix, "(E)")

    def test_empty_input(self):
        proj_id, class_suffix = resolve_fund_proj_id_and_class("", self.registry)
        self.assertIsNone(proj_id)
        self.assertIsNone(class_suffix)

    def test_none_input(self):
        proj_id, class_suffix = resolve_fund_proj_id_and_class(None, self.registry)
        self.assertIsNone(proj_id)
        self.assertIsNone(class_suffix)

    def test_unresolvable(self):
        proj_id, class_suffix = resolve_fund_proj_id_and_class("UNKNOWN_FUND", self.registry)
        self.assertIsNone(proj_id)

    def test_dash_class_suffix(self):
        proj_id, class_suffix = resolve_fund_proj_id_and_class("SCBNDQ-SSF", self.registry)
        self.assertEqual(class_suffix, "(SSF)")

    def test_fund_suffix_auto_added(self):
        # SCBS&P500 → SCBS&P500FUND should match
        proj_id, _ = resolve_fund_proj_id_and_class("SCBS&P500", self.registry)
        self.assertEqual(proj_id, "PROJ_002")


class TestGetSecApiKeys(unittest.TestCase):
    def test_from_provided_keys(self):
        keys = get_sec_api_keys(["key1", "key2"])
        self.assertEqual(keys, ["key1", "key2"])

    def test_filters_empty(self):
        keys = get_sec_api_keys(["key1", "", None])
        self.assertNotIn("", keys)

    def test_no_input_no_env(self):
        keys = get_sec_api_keys(None)
        # Only env vars, which may or may not be set
        self.assertIsInstance(keys, list)


class TestExtractNav(unittest.TestCase):
    def test_list_with_single_item(self):
        data = [{"last_val": 13.5, "previous_val": 13.4}]
        last, prev = _extract_nav(data, None)
        self.assertEqual(last, 13.5)
        self.assertEqual(prev, 13.4)

    def test_list_with_class_match(self):
        data = [
            {"class_abbr_name": "SCBNDQ(A)", "last_val": 14.0, "previous_val": 13.8},
            {"class_abbr_name": "SCBNDQ(E)", "last_val": 13.5, "previous_val": 13.4},
        ]
        last, prev = _extract_nav(data, "(E)")
        self.assertEqual(last, 13.5)
        self.assertEqual(prev, 13.4)

    def test_list_no_class_match_falls_back(self):
        data = [{"last_val": 14.0, "previous_val": 13.8, "class_abbr_name": "SCBNDQ(A)"}]
        last, prev = _extract_nav(data, "(X)")
        self.assertEqual(last, 14.0)  # falls back to first item

    def test_dict_format(self):
        data = {"last_val": 13.5, "previous_val": 13.4}
        last, prev = _extract_nav(data, None)
        self.assertEqual(last, 13.5)

    def test_empty_list(self):
        last, prev = _extract_nav([], None)
        self.assertIsNone(last)
        self.assertIsNone(prev)


class TestCallSecApi(unittest.TestCase):
    """Tests for call_sec_api — rate-limited SEC API caller with key fallback."""

    @patch("sniper.funds.requests.get")
    def test_returns_response_on_200(self, mock_get):
        resp = MagicMock(status_code=200)
        mock_get.return_value = resp
        result = call_sec_api("https://example.com/api", api_keys=["key1"])
        self.assertEqual(result, resp)
        mock_get.assert_called_once()
        headers = mock_get.call_args[1]["headers"]
        self.assertEqual(headers["Ocp-Apim-Subscription-Key"], "key1")

    @patch("sniper.funds.requests.get")
    def test_returns_none_when_no_keys(self, mock_get):
        result = call_sec_api("https://example.com/api", api_keys=[])
        self.assertIsNone(result)
        mock_get.assert_not_called()

    @patch("sniper.funds.requests.get")
    def test_falls_back_to_second_key_on_non_200(self, mock_get):
        bad_resp = MagicMock(status_code=403)
        good_resp = MagicMock(status_code=200)
        mock_get.side_effect = [bad_resp, good_resp]
        result = call_sec_api("https://example.com/api", api_keys=["bad", "good"])
        self.assertEqual(result, good_resp)
        self.assertEqual(mock_get.call_count, 2)

    @patch("sniper.funds.requests.get")
    def test_returns_none_when_all_keys_fail(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        result = call_sec_api("https://example.com/api", api_keys=["k1", "k2"])
        self.assertIsNone(result)
        self.assertEqual(mock_get.call_count, 2)

    @patch("sniper.funds.requests.get", side_effect=Exception("network error"))
    def test_handles_exception_gracefully(self, mock_get):
        result = call_sec_api("https://example.com/api", api_keys=["k1"])
        self.assertIsNone(result)

    @patch("sniper.funds.get_sec_api_keys", return_value=["env_key"])
    @patch("sniper.funds.requests.get")
    def test_uses_default_keys_when_none_passed(self, mock_get, mock_keys):
        mock_get.return_value = MagicMock(status_code=200)
        call_sec_api("https://example.com/api", api_keys=None)
        mock_keys.assert_called_once()

    @patch("sniper.funds.requests.get")
    def test_exception_on_first_key_tries_second(self, mock_get):
        good_resp = MagicMock(status_code=200)
        mock_get.side_effect = [Exception("timeout"), good_resp]
        result = call_sec_api("https://example.com/api", api_keys=["bad", "good"])
        self.assertEqual(result, good_resp)


class TestLoadDiskFundRegistry(unittest.TestCase):
    """Tests for load_disk_fund_registry — disk cache loader."""

    @patch("sniper.funds._FUND_REGISTRY_CACHE_PATH")
    def test_returns_cached_registry_when_fresh(self, mock_path):
        registry = {"SCBNDQ": "PROJ_001"}
        cache_data = {"ts": time.time(), "registry": registry}
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = json.dumps(cache_data)
        result = load_disk_fund_registry()
        self.assertEqual(result, registry)

    @patch("sniper.funds._FUND_REGISTRY_CACHE_PATH")
    def test_returns_none_when_cache_expired(self, mock_path):
        registry = {"SCBNDQ": "PROJ_001"}
        cache_data = {"ts": time.time() - 200000, "registry": registry}
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = json.dumps(cache_data)
        result = load_disk_fund_registry()
        self.assertIsNone(result)

    @patch("sniper.funds._FUND_REGISTRY_CACHE_PATH")
    def test_returns_none_when_no_cache_file(self, mock_path):
        mock_path.exists.return_value = False
        result = load_disk_fund_registry()
        self.assertIsNone(result)

    @patch("sniper.funds._FUND_REGISTRY_CACHE_PATH")
    def test_returns_none_when_empty_registry(self, mock_path):
        cache_data = {"ts": time.time(), "registry": {}}
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = json.dumps(cache_data)
        result = load_disk_fund_registry()
        self.assertIsNone(result)

    @patch("sniper.funds._FUND_REGISTRY_CACHE_PATH")
    def test_returns_none_on_json_decode_error(self, mock_path):
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "not valid json"
        result = load_disk_fund_registry()
        self.assertIsNone(result)

    @patch("sniper.funds._FUND_REGISTRY_CACHE_PATH")
    def test_returns_none_when_ts_missing(self, mock_path):
        cache_data = {"registry": {"SCBNDQ": "PROJ_001"}}
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = json.dumps(cache_data)
        # ts defaults to 0, so time.time() - 0 > max_age → expired
        result = load_disk_fund_registry()
        self.assertIsNone(result)


class TestSaveDiskFundRegistry(unittest.TestCase):
    """Tests for save_disk_fund_registry — disk cache writer."""

    @patch("sniper.funds._FUND_REGISTRY_CACHE_PATH")
    def test_writes_registry_with_timestamp(self, mock_path):
        mock_parent = MagicMock()
        mock_path.parent = mock_parent
        registry = {"SCBNDQ": "PROJ_001"}
        save_disk_fund_registry(registry)
        mock_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_path.write_text.assert_called_once()
        written = json.loads(mock_path.write_text.call_args[0][0])
        self.assertIn("ts", written)
        self.assertEqual(written["registry"], registry)

    @patch("sniper.funds._FUND_REGISTRY_CACHE_PATH")
    def test_handles_write_error_gracefully(self, mock_path):
        mock_path.parent = MagicMock()
        mock_path.write_text.side_effect = PermissionError("no write")
        # Should not raise
        save_disk_fund_registry({"SCBNDQ": "PROJ_001"})

    @patch("sniper.funds._FUND_REGISTRY_CACHE_PATH")
    def test_handles_empty_registry(self, mock_path):
        mock_path.parent = MagicMock()
        save_disk_fund_registry({})
        mock_path.write_text.assert_called_once()
        written = json.loads(mock_path.write_text.call_args[0][0])
        self.assertEqual(written["registry"], {})


class TestBuildFundRegistry(unittest.TestCase):
    """Tests for build_fund_registry — multi-API registry orchestration."""

    @patch("sniper.funds.save_disk_fund_registry")
    @patch("sniper.funds.call_sec_api")
    @patch("sniper.funds.load_disk_fund_registry")
    def test_returns_cached_if_available(self, mock_load, mock_api, mock_save):
        cached = {"SCBNDQ": "PROJ_001"}
        mock_load.return_value = cached
        result = build_fund_registry(api_keys=["k1"])
        self.assertEqual(result, cached)
        mock_api.assert_not_called()
        mock_save.assert_not_called()

    @patch("sniper.funds.save_disk_fund_registry")
    @patch("sniper.funds.call_sec_api")
    @patch("sniper.funds.load_disk_fund_registry", return_value=None)
    def test_builds_registry_from_api(self, mock_load, mock_api, mock_save):
        amc_resp = MagicMock(status_code=200)
        amc_resp.content = json.dumps([{"unique_id": "AMC001"}]).encode()
        fund_resp = MagicMock(status_code=200)
        fund_resp.content = json.dumps([
            {"proj_abbr_name": "SCBNDQ", "proj_id": "PROJ_001"},
            {"proj_abbr_name": "SCBS&P500", "proj_id": "PROJ_002"},
        ]).encode()
        mock_api.side_effect = [amc_resp, fund_resp]

        result = build_fund_registry(api_keys=["k1"])
        self.assertEqual(result, {"SCBNDQ": "PROJ_001", "SCBS&P500": "PROJ_002"})
        mock_save.assert_called_once_with(result)

    @patch("sniper.funds.save_disk_fund_registry")
    @patch("sniper.funds.call_sec_api", return_value=None)
    @patch("sniper.funds.load_disk_fund_registry", return_value=None)
    def test_returns_empty_when_amc_api_fails(self, mock_load, mock_api, mock_save):
        result = build_fund_registry(api_keys=["k1"])
        self.assertEqual(result, {})
        mock_save.assert_not_called()

    @patch("sniper.funds.save_disk_fund_registry")
    @patch("sniper.funds.call_sec_api")
    @patch("sniper.funds.load_disk_fund_registry", return_value=None)
    def test_skips_amcs_without_unique_id(self, mock_load, mock_api, mock_save):
        amc_resp = MagicMock(status_code=200)
        amc_resp.content = json.dumps([{"name": "No ID AMC"}]).encode()
        mock_api.return_value = amc_resp

        result = build_fund_registry(api_keys=["k1"])
        self.assertEqual(result, {})
        # Only 1 call: the AMC list; no fund calls because unique_id missing
        self.assertEqual(mock_api.call_count, 1)

    @patch("sniper.funds.save_disk_fund_registry")
    @patch("sniper.funds.call_sec_api")
    @patch("sniper.funds.load_disk_fund_registry", return_value=None)
    def test_skips_funds_without_abbr_or_proj_id(self, mock_load, mock_api, mock_save):
        amc_resp = MagicMock(status_code=200)
        amc_resp.content = json.dumps([{"unique_id": "AMC001"}]).encode()
        fund_resp = MagicMock(status_code=200)
        fund_resp.content = json.dumps([
            {"proj_abbr_name": "SCBNDQ"},  # missing proj_id
            {"proj_id": "PROJ_002"},  # missing abbr
        ]).encode()
        mock_api.side_effect = [amc_resp, fund_resp]

        result = build_fund_registry(api_keys=["k1"])
        self.assertEqual(result, {})
        mock_save.assert_not_called()

    @patch("sniper.funds.save_disk_fund_registry")
    @patch("sniper.funds.call_sec_api")
    @patch("sniper.funds.load_disk_fund_registry", return_value=None)
    def test_handles_fund_api_returning_none(self, mock_load, mock_api, mock_save):
        amc_resp = MagicMock(status_code=200)
        amc_resp.content = json.dumps([{"unique_id": "AMC001"}]).encode()
        mock_api.side_effect = [amc_resp, None]

        result = build_fund_registry(api_keys=["k1"])
        self.assertEqual(result, {})

    @patch("sniper.funds.save_disk_fund_registry")
    @patch("sniper.funds.call_sec_api")
    @patch("sniper.funds.load_disk_fund_registry", return_value=None)
    def test_handles_malformed_fund_json(self, mock_load, mock_api, mock_save):
        amc_resp = MagicMock(status_code=200)
        amc_resp.content = json.dumps([{"unique_id": "AMC001"}]).encode()
        fund_resp = MagicMock(status_code=200)
        fund_resp.content = b"not json"
        mock_api.side_effect = [amc_resp, fund_resp]

        result = build_fund_registry(api_keys=["k1"])
        self.assertEqual(result, {})


class TestFetchFundNavWithPrevious(unittest.TestCase):
    """Tests for fetch_fund_nav_with_previous — multi-date NAV fallback."""

    @patch("sniper.funds.call_sec_api")
    def test_returns_nav_with_previous_from_single_date(self, mock_api):
        resp = MagicMock(status_code=200)
        resp.content = json.dumps([{"last_val": 13.5, "previous_val": 13.4}]).encode()
        mock_api.return_value = resp

        last, prev = fetch_fund_nav_with_previous("PROJ_001", api_keys=["k1"])
        self.assertEqual(last, 13.5)
        self.assertEqual(prev, 13.4)

    @patch("sniper.funds.call_sec_api")
    def test_returns_zero_when_api_returns_none(self, mock_api):
        mock_api.return_value = None
        last, prev = fetch_fund_nav_with_previous("PROJ_001", api_keys=["k1"])
        self.assertEqual(last, 0.0)
        self.assertEqual(prev, 0.0)

    @patch("sniper.funds.call_sec_api")
    def test_falls_back_to_two_dates_when_previous_is_zero(self, mock_api):
        resp1 = MagicMock(status_code=200)
        resp1.content = json.dumps([{"last_val": 14.0, "previous_val": 0}]).encode()
        resp2 = MagicMock(status_code=200)
        resp2.content = json.dumps([{"last_val": 13.5, "previous_val": 0}]).encode()
        mock_api.side_effect = [resp1, resp2]

        last, prev = fetch_fund_nav_with_previous("PROJ_001", api_keys=["k1"])
        self.assertEqual(last, 14.0)
        self.assertEqual(prev, 13.5)

    @patch("sniper.funds.call_sec_api")
    def test_returns_single_nav_with_zero_prev_if_only_one_date(self, mock_api):
        resp = MagicMock(status_code=200)
        resp.content = json.dumps([{"last_val": 14.0, "previous_val": ""}]).encode()
        # Return data for first call, then None for the rest
        mock_api.side_effect = [resp] + [None] * 10

        last, prev = fetch_fund_nav_with_previous("PROJ_001", api_keys=["k1"])
        self.assertEqual(last, 14.0)
        self.assertEqual(prev, 0.0)

    @patch("sniper.funds.call_sec_api")
    def test_returns_zeros_when_no_last_val(self, mock_api):
        resp = MagicMock(status_code=200)
        resp.content = json.dumps([{"last_val": None, "previous_val": None}]).encode()
        mock_api.return_value = resp

        last, prev = fetch_fund_nav_with_previous("PROJ_001", api_keys=["k1"])
        self.assertEqual(last, 0.0)
        self.assertEqual(prev, 0.0)

    @patch("sniper.funds.call_sec_api")
    def test_with_fund_class_filter(self, mock_api):
        resp = MagicMock(status_code=200)
        resp.content = json.dumps([
            {"class_abbr_name": "FUND(A)", "last_val": 10.0, "previous_val": 9.8},
            {"class_abbr_name": "FUND(E)", "last_val": 13.5, "previous_val": 13.4},
        ]).encode()
        mock_api.return_value = resp

        last, prev = fetch_fund_nav_with_previous("PROJ_001", fund_class="(E)", api_keys=["k1"])
        self.assertEqual(last, 13.5)
        self.assertEqual(prev, 13.4)

    @patch("sniper.funds.call_sec_api")
    def test_previous_val_zero_string_triggers_fallback(self, mock_api):
        resp1 = MagicMock(status_code=200)
        resp1.content = json.dumps([{"last_val": 14.0, "previous_val": "0"}]).encode()
        resp2 = MagicMock(status_code=200)
        resp2.content = json.dumps([{"last_val": 13.0, "previous_val": "0"}]).encode()
        mock_api.side_effect = [resp1, resp2]

        last, prev = fetch_fund_nav_with_previous("PROJ_001", api_keys=["k1"])
        self.assertEqual(last, 14.0)
        self.assertEqual(prev, 13.0)

    @patch("sniper.funds.call_sec_api")
    def test_non_200_status_code_skipped(self, mock_api):
        bad_resp = MagicMock(status_code=404)
        good_resp = MagicMock(status_code=200)
        good_resp.content = json.dumps([{"last_val": 12.0, "previous_val": 11.9}]).encode()
        mock_api.side_effect = [bad_resp, good_resp]

        last, prev = fetch_fund_nav_with_previous("PROJ_001", api_keys=["k1"])
        self.assertEqual(last, 12.0)
        self.assertEqual(prev, 11.9)


class TestFetchFundNav(unittest.TestCase):
    """Tests for fetch_fund_nav — simple wrapper returning only last NAV."""

    @patch("sniper.funds.fetch_fund_nav_with_previous", return_value=(13.5, 13.4))
    def test_returns_last_val(self, mock_fn):
        result = fetch_fund_nav("PROJ_001", "(E)", api_keys=["k1"])
        self.assertEqual(result, 13.5)
        mock_fn.assert_called_once_with("PROJ_001", "(E)", api_keys=["k1"])

    @patch("sniper.funds.fetch_fund_nav_with_previous", return_value=(0.0, 0.0))
    def test_returns_zero_on_failure(self, mock_fn):
        result = fetch_fund_nav("PROJ_001")
        self.assertEqual(result, 0.0)


class TestFetchMasterTrends(unittest.TestCase):
    """Tests for fetch_master_trends — yfinance day-change fetch."""

    @patch("sniper.funds.yf.download")
    def test_single_symbol(self, mock_dl):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        close = pd.Series([100.0, 105.0], index=idx, name="Close")
        df = pd.DataFrame({"Close": close})
        mock_dl.return_value = df

        result = fetch_master_trends(["SPY"])
        self.assertAlmostEqual(result["SPY"], 5.0)

    @patch("sniper.funds.yf.download")
    def test_multiple_symbols(self, mock_dl):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        close_df = pd.DataFrame({
            "QQQ": [200.0, 210.0],
            "SPY": [100.0, 105.0],
        }, index=idx)
        # yf.download for multiple symbols returns MultiIndex columns
        multi = pd.concat({"Close": close_df}, axis=1)
        mock_dl.return_value = multi

        result = fetch_master_trends(["SPY", "QQQ"])
        self.assertAlmostEqual(result["SPY"], 5.0)
        self.assertAlmostEqual(result["QQQ"], 5.0)

    @patch("sniper.funds.yf.download")
    def test_empty_symbols(self, mock_dl):
        result = fetch_master_trends([])
        self.assertEqual(result, {})
        mock_dl.assert_not_called()

    @patch("sniper.funds.yf.download")
    def test_filters_na_symbols(self, mock_dl):
        result = fetch_master_trends(["N/A", "", " "])
        self.assertEqual(result, {})
        mock_dl.assert_not_called()

    @patch("sniper.funds.yf.download", side_effect=Exception("API error"))
    def test_handles_yfinance_exception(self, mock_dl):
        result = fetch_master_trends(["SPY"])
        self.assertEqual(result, {"SPY": 0.0})

    @patch("sniper.funds.yf.download")
    def test_returns_zero_when_download_empty(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        result = fetch_master_trends(["SPY"])
        self.assertEqual(result, {"SPY": 0.0})

    @patch("sniper.funds.yf.download")
    def test_deduplicates_and_uppercases(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        result = fetch_master_trends(["spy", "SPY", " spy "])
        self.assertIn("SPY", result)
        self.assertEqual(len(result), 1)

    @patch("sniper.funds.yf.download")
    def test_zero_previous_close_gives_zero_change(self, mock_dl):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        close = pd.Series([0.0, 105.0], index=idx, name="Close")
        df = pd.DataFrame({"Close": close})
        mock_dl.return_value = df

        result = fetch_master_trends(["SPY"])
        self.assertEqual(result["SPY"], 0.0)


class TestGetMasterData(unittest.TestCase):
    """Tests for get_master_data — build master ETF data from fund list."""

    @patch("sniper.funds.fetch_master_trends")
    def test_extracts_masters_and_delegates(self, mock_trends):
        mock_trends.return_value = {"SPY": 1.5, "QQQ": 2.0}
        fund_list = [
            {"Code": "F1", "Master": "SPY"},
            {"Code": "F2", "Master": "QQQ"},
        ]
        result = get_master_data(fund_list)
        self.assertEqual(result, {"SPY": 1.5, "QQQ": 2.0})

    @patch("sniper.funds.fetch_master_trends")
    def test_returns_empty_when_no_masters(self, mock_trends):
        fund_list = [
            {"Code": "F1"},
            {"Code": "F2", "Master": "N/A"},
        ]
        result = get_master_data(fund_list)
        self.assertEqual(result, {})
        mock_trends.assert_not_called()

    @patch("sniper.funds.fetch_master_trends")
    def test_filters_na_masters(self, mock_trends):
        mock_trends.return_value = {"SPY": 1.0}
        fund_list = [
            {"Code": "F1", "Master": "SPY"},
            {"Code": "F2", "Master": "N/A"},
            {"Code": "F3"},
        ]
        get_master_data(fund_list)
        # Only "SPY" passed
        args = mock_trends.call_args[0][0]
        self.assertIn("SPY", args)

    def test_empty_fund_list(self):
        result = get_master_data([])
        self.assertEqual(result, {})


class TestGetFundNavByCode(unittest.TestCase):
    """Tests for get_fund_nav_by_code — registry NAV lookup with class fallbacks."""

    @patch("sniper.funds.fetch_fund_nav")
    def test_returns_nav_with_class_suffix(self, mock_nav):
        mock_nav.return_value = 13.5
        registry = {"SCBNDQ": "PROJ_001"}
        result = get_fund_nav_by_code("SCBNDQ(E)", registry, api_keys=["k1"])
        self.assertEqual(result, 13.5)
        mock_nav.assert_called_once_with("PROJ_001", "(E)", api_keys=["k1"])

    @patch("sniper.funds.fetch_fund_nav")
    def test_falls_back_to_no_class(self, mock_nav):
        # First call with class returns 0, second without class returns nav
        mock_nav.side_effect = [0.0, 15.0]
        registry = {"SCBNDQ": "PROJ_001"}
        result = get_fund_nav_by_code("SCBNDQ(E)", registry, api_keys=["k1"])
        self.assertEqual(result, 15.0)
        self.assertEqual(mock_nav.call_count, 2)

    @patch("sniper.funds.fetch_fund_nav")
    def test_falls_back_to_alt_classes(self, mock_nav):
        # First call with class: 0, second without class: 0, then alt classes
        alt_classes = ['(E)', '(SSF)', '(SSFE)', '(SSFA)', '(SSFX)', '(A)']
        # 0 for original class, 0 for None, then 0 for first 2 alts, then success
        mock_nav.side_effect = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0]
        registry = {"SCBNDQ": "PROJ_001"}
        result = get_fund_nav_by_code("SCBNDQ(E)", registry, api_keys=["k1"])
        self.assertEqual(result, 20.0)

    @patch("sniper.funds.fetch_fund_nav")
    def test_returns_zero_when_unresolvable(self, mock_nav):
        registry = {"SCBNDQ": "PROJ_001"}
        result = get_fund_nav_by_code("UNKNOWN", registry, api_keys=["k1"])
        self.assertEqual(result, 0.0)
        mock_nav.assert_not_called()

    @patch("sniper.funds.fetch_fund_nav", return_value=0.0)
    def test_returns_zero_when_all_fallbacks_fail(self, mock_nav):
        registry = {"SCBNDQ": "PROJ_001"}
        result = get_fund_nav_by_code("SCBNDQ", registry, api_keys=["k1"])
        self.assertEqual(result, 0.0)
        # 1 (original) + 1 (None) + 6 (alt classes) = 8 calls
        self.assertEqual(mock_nav.call_count, 8)


class TestGetFundData(unittest.TestCase):
    """Tests for get_fund_data — ThreadPoolExecutor-based fund assembly."""

    def _make_fund(self, code="SCBNDQ(E)", units=100.0, cost=10.0, master="SPY"):
        return {"Code": code, "Units": units, "Cost": cost, "Master": master}

    def test_basic_fund_data_assembly(self):
        fund_list = [self._make_fund()]
        registry = {"SCBNDQ": "PROJ_001"}
        master_trends = {"SPY": 1.5}
        fetch_nav_fn = MagicMock(return_value=(13.5, 13.0))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, master_trends, fetch_nav_fn, get_nav_fn)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["Last Price"], 13.5)
        self.assertAlmostEqual(df.iloc[0]["Value"], 100.0 * 13.5)
        self.assertAlmostEqual(df.iloc[0]["Cost Basis"], 100.0 * 10.0)
        expected_pl = (100.0 * 13.5) - (100.0 * 10.0)
        self.assertAlmostEqual(df.iloc[0]["P/L"], expected_pl)

    def test_day_gain_calculation(self):
        fund_list = [self._make_fund()]
        registry = {"SCBNDQ": "PROJ_001"}
        master_trends = {"SPY": 1.0}
        fetch_nav_fn = MagicMock(return_value=(105.0, 100.0))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, master_trends, fetch_nav_fn, get_nav_fn)
        self.assertAlmostEqual(df.iloc[0]["Fund Day Gain %"], 5.0)
        self.assertAlmostEqual(df.iloc[0]["Master Day Gain %"], 1.0)
        self.assertAlmostEqual(df.iloc[0]["Master vs Fund %"], 4.0)

    def test_falls_back_to_nav_by_code_fn_when_nav_zero(self):
        fund_list = [self._make_fund()]
        registry = {"SCBNDQ": "PROJ_001"}
        master_trends = {"SPY": 0.0}
        fetch_nav_fn = MagicMock(return_value=(0.0, 0.0))
        get_nav_fn = MagicMock(return_value=15.0)

        df = get_fund_data(fund_list, registry, master_trends, fetch_nav_fn, get_nav_fn)
        self.assertAlmostEqual(df.iloc[0]["Last Price"], 15.0)
        get_nav_fn.assert_called_once()

    def test_falls_back_to_cost_when_all_nav_zero(self):
        fund_list = [self._make_fund(cost=10.0)]
        registry = {"SCBNDQ": "PROJ_001"}
        master_trends = {}
        fetch_nav_fn = MagicMock(return_value=(0.0, 0.0))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, master_trends, fetch_nav_fn, get_nav_fn)
        self.assertAlmostEqual(df.iloc[0]["Last Price"], 10.0)

    def test_empty_fund_list(self):
        df = get_fund_data([], {}, {}, MagicMock(), MagicMock())
        self.assertEqual(len(df), 0)
        self.assertIn("Code", df.columns)
        self.assertIn("P/L %", df.columns)

    def test_na_master_ticker(self):
        fund_list = [self._make_fund(master="N/A")]
        registry = {"SCBNDQ": "PROJ_001"}
        master_trends = {}
        fetch_nav_fn = MagicMock(return_value=(13.5, 13.0))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, master_trends, fetch_nav_fn, get_nav_fn)
        self.assertEqual(df.iloc[0]["Master"], "N/A")
        self.assertEqual(df.iloc[0]["Master Day Gain %"], 0.0)

    def test_no_master_key(self):
        fund_list = [{"Code": "SCBNDQ(E)", "Units": 100.0, "Cost": 10.0}]
        registry = {"SCBNDQ": "PROJ_001"}
        fetch_nav_fn = MagicMock(return_value=(13.0, 12.5))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, {}, fetch_nav_fn, get_nav_fn)
        self.assertEqual(df.iloc[0]["Master"], "N/A")

    def test_previous_price_none_when_same_as_nav(self):
        fund_list = [self._make_fund()]
        registry = {"SCBNDQ": "PROJ_001"}
        # prev_nav equals current_nav → Previous Price should be None
        fetch_nav_fn = MagicMock(return_value=(13.5, 13.5))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, {"SPY": 0.0}, fetch_nav_fn, get_nav_fn)
        self.assertIsNone(df.iloc[0]["Previous Price"])

    def test_previous_price_none_when_zero(self):
        fund_list = [self._make_fund()]
        registry = {"SCBNDQ": "PROJ_001"}
        fetch_nav_fn = MagicMock(return_value=(13.5, 0.0))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, {"SPY": 0.0}, fetch_nav_fn, get_nav_fn)
        self.assertIsNone(df.iloc[0]["Previous Price"])

    def test_multiple_funds(self):
        fund_list = [
            self._make_fund(code="SCBNDQ(E)", units=100.0, cost=10.0, master="SPY"),
            self._make_fund(code="SCBNDQ(A)", units=50.0, cost=20.0, master="QQQ"),
        ]
        registry = {"SCBNDQ": "PROJ_001"}
        master_trends = {"SPY": 1.0, "QQQ": 2.0}
        fetch_nav_fn = MagicMock(return_value=(15.0, 14.0))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, master_trends, fetch_nav_fn, get_nav_fn)
        self.assertEqual(len(df), 2)

    def test_output_columns(self):
        fund_list = [self._make_fund()]
        registry = {"SCBNDQ": "PROJ_001"}
        fetch_nav_fn = MagicMock(return_value=(13.5, 13.0))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, {"SPY": 0.0}, fetch_nav_fn, get_nav_fn)
        expected_cols = [
            'Code', 'Units', 'Cost', 'Last Price', 'Previous Price',
            'Fund Day Gain %', 'Master', 'Master Day Gain %',
            'Master vs Fund %', 'Cost Basis', 'Value', 'P/L', 'P/L %',
        ]
        self.assertEqual(list(df.columns), expected_cols)

    def test_unresolvable_code_gets_zero_nav(self):
        fund_list = [self._make_fund(code="UNKNOWN")]
        registry = {}
        fetch_nav_fn = MagicMock(return_value=(0.0, 0.0))
        get_nav_fn = MagicMock(return_value=0.0)

        df = get_fund_data(fund_list, registry, {}, fetch_nav_fn, get_nav_fn)
        # Falls back to cost
        self.assertAlmostEqual(df.iloc[0]["Last Price"], 10.0)


if __name__ == "__main__":
    unittest.main()
