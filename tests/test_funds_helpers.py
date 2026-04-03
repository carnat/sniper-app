"""Tests for sniper/funds.py — fund data helpers (pure functions only)."""

import unittest

from sniper.funds import (
    normalize_fund_token,
    resolve_fund_proj_id_and_class,
    get_sec_api_keys,
    _extract_nav,
    FUND_API_MAPPING,
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


if __name__ == "__main__":
    unittest.main()
