"""
tests/test_gates.py — 17 unit tests for lib/gates.py

Coverage:
 - classify_vix: GREEN / CAUTION / FREEZE / None
 - vix_zone_label: text content
 - classify_fx: GO / CAUTION / None / boundary values
 - fx_zone_label: zone A / B / C
 - check_volume_gate: normal / spike / low / missing data
 - check_earnings_gate: clear / blackout / today / unknown
 - run_gates: aggregation, worst-signal precedence
"""

import sys
import unittest
from pathlib import Path

# Ensure lib/ is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.gates import (
    GateResult,
    RunResult,
    check_earnings_gate,
    check_volume_gate,
    classify_fx,
    classify_vix,
    fx_zone_label,
    run_gates,
    vix_zone_label,
)
from lib.constants import (
    FX_ZONE_A,
    FX_ZONE_C,
    GATE_SIGNALS,
    VIX_CAUTION,
    VIX_FREEZE,
)


class TestClassifyVix(unittest.TestCase):
    def test_below_caution_is_go(self):
        self.assertEqual(classify_vix(VIX_CAUTION - 0.1), GATE_SIGNALS["GO"])

    def test_at_caution_is_caution(self):
        self.assertEqual(classify_vix(VIX_CAUTION), GATE_SIGNALS["CAUTION"])

    def test_between_caution_and_freeze_is_caution(self):
        self.assertEqual(classify_vix(VIX_FREEZE - 0.1), GATE_SIGNALS["CAUTION"])

    def test_at_freeze_is_freeze(self):
        self.assertEqual(classify_vix(VIX_FREEZE), GATE_SIGNALS["FREEZE"])

    def test_above_freeze_is_freeze(self):
        self.assertEqual(classify_vix(VIX_FREEZE + 10), GATE_SIGNALS["FREEZE"])

    def test_none_is_caution(self):
        self.assertEqual(classify_vix(None), GATE_SIGNALS["CAUTION"])


class TestVixZoneLabel(unittest.TestCase):
    def test_green_label_contains_green(self):
        label = vix_zone_label(15.0)
        self.assertIn("GREEN", label)

    def test_caution_label_contains_caution(self):
        label = vix_zone_label(22.0)
        self.assertIn("CAUTION", label)

    def test_freeze_label_contains_freeze(self):
        label = vix_zone_label(30.0)
        self.assertIn("FREEZE", label)

    def test_none_returns_unknown(self):
        self.assertEqual(vix_zone_label(None), "UNKNOWN")


class TestClassifyFx(unittest.TestCase):
    def test_zone_a_is_go(self):
        self.assertEqual(classify_fx(FX_ZONE_A), GATE_SIGNALS["GO"])

    def test_zone_b_is_go(self):
        mid = (FX_ZONE_A + FX_ZONE_C) / 2
        self.assertEqual(classify_fx(mid), GATE_SIGNALS["GO"])

    def test_at_zone_c_is_caution(self):
        self.assertEqual(classify_fx(FX_ZONE_C), GATE_SIGNALS["CAUTION"])

    def test_above_zone_c_is_caution(self):
        self.assertEqual(classify_fx(FX_ZONE_C + 2), GATE_SIGNALS["CAUTION"])

    def test_none_is_caution(self):
        self.assertEqual(classify_fx(None), GATE_SIGNALS["CAUTION"])


class TestFxZoneLabel(unittest.TestCase):
    def test_dinner_signal_in_zone_a(self):
        label = fx_zone_label(FX_ZONE_A)
        self.assertIn("Zone A", label)
        self.assertIn("🍽️", label)

    def test_neutral_in_zone_b(self):
        label = fx_zone_label(34.0)
        self.assertIn("Zone B", label)

    def test_hold_in_zone_c(self):
        label = fx_zone_label(FX_ZONE_C)
        self.assertIn("Zone C", label)


class TestVolumeGate(unittest.TestCase):
    def test_normal_volume_is_go(self):
        result = check_volume_gate(today_volume=1_000_000, adv_20=1_000_000)
        self.assertEqual(result.signal, GATE_SIGNALS["GO"])
        self.assertEqual(result.gate, "Q3_VOLUME")

    def test_spike_is_caution(self):
        result = check_volume_gate(today_volume=6_000_000, adv_20=1_000_000)
        self.assertEqual(result.signal, GATE_SIGNALS["CAUTION"])
        self.assertGreater(result.value, 5.0)

    def test_thin_volume_is_caution(self):
        result = check_volume_gate(today_volume=200_000, adv_20=1_000_000)
        self.assertEqual(result.signal, GATE_SIGNALS["CAUTION"])
        self.assertLess(result.value, 0.3)

    def test_missing_data_is_caution(self):
        result = check_volume_gate(today_volume=None, adv_20=None)
        self.assertEqual(result.signal, GATE_SIGNALS["CAUTION"])


class TestEarningsGate(unittest.TestCase):
    def test_far_earnings_is_go(self):
        result = check_earnings_gate(days_to_earnings=30)
        self.assertEqual(result.signal, GATE_SIGNALS["GO"])

    def test_inside_blackout_is_caution(self):
        result = check_earnings_gate(days_to_earnings=3, blackout_days=5)
        self.assertEqual(result.signal, GATE_SIGNALS["CAUTION"])

    def test_at_blackout_boundary_is_caution(self):
        result = check_earnings_gate(days_to_earnings=5, blackout_days=5)
        self.assertEqual(result.signal, GATE_SIGNALS["CAUTION"])

    def test_earnings_today_is_caution(self):
        result = check_earnings_gate(days_to_earnings=0)
        self.assertEqual(result.signal, GATE_SIGNALS["CAUTION"])

    def test_unknown_earnings_is_go(self):
        result = check_earnings_gate(days_to_earnings=None)
        self.assertEqual(result.signal, GATE_SIGNALS["GO"])


class TestRunGates(unittest.TestCase):
    def test_all_clear_returns_go(self):
        result = run_gates(
            ticker="AAPL",
            vix=15.0,
            usd_thb=33.0,
            today_volume=1_000_000,
            adv_20=1_000_000,
            days_to_earnings=30,
        )
        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.signal, GATE_SIGNALS["GO"])
        self.assertTrue(result.is_go)
        self.assertEqual(len(result.gates), 4)

    def test_high_vix_returns_freeze(self):
        result = run_gates(
            ticker="AAPL",
            vix=30.0,
            usd_thb=33.0,
            today_volume=1_000_000,
            adv_20=1_000_000,
            days_to_earnings=30,
        )
        self.assertEqual(result.signal, GATE_SIGNALS["FREEZE"])
        self.assertTrue(result.is_freeze)

    def test_fx_caution_returns_caution_when_vix_ok(self):
        result = run_gates(
            ticker="MU",
            vix=15.0,
            usd_thb=37.0,
            today_volume=1_000_000,
            adv_20=1_000_000,
            days_to_earnings=30,
        )
        self.assertEqual(result.signal, GATE_SIGNALS["CAUTION"])

    def test_freeze_beats_caution(self):
        """FREEZE on VIX must win over CAUTION on FX."""
        result = run_gates(
            ticker="TSM",
            vix=30.0,   # FREEZE
            usd_thb=37.0,  # CAUTION
        )
        self.assertEqual(result.signal, GATE_SIGNALS["FREEZE"])

    def test_gates_list_has_correct_identifiers(self):
        result = run_gates(ticker="ASTS", vix=15.0, usd_thb=33.0)
        gate_ids = [g.gate for g in result.gates]
        self.assertIn("Q1_VIX", gate_ids)
        self.assertIn("Q2_FX", gate_ids)
        self.assertIn("Q3_VOLUME", gate_ids)
        self.assertIn("Q4_EARNINGS", gate_ids)


if __name__ == "__main__":
    unittest.main()
