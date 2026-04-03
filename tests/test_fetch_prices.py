"""Tests for scripts/fetch_prices.py — regime and zone helpers."""

import sys
import unittest
from pathlib import Path

# Add parent directory for script imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_prices import get_regime, get_thb_zone


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


if __name__ == "__main__":
    unittest.main()
