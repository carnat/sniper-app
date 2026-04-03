"""Tests for sniper/version.py — app version management."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sniper.version import get_app_version


class TestGetAppVersion(unittest.TestCase):
    def test_reads_version_file(self):
        version = get_app_version()
        # VERSION file exists in the repo; should not be fallback
        self.assertNotEqual(version, "0.0.0")
        self.assertRegex(version, r"^\d+\.\d+\.\d+")

    def test_fallback_when_file_missing(self):
        with patch("sniper.version.Path") as mock_path_cls:
            mock_file = mock_path_cls.return_value.resolve.return_value.parent.parent.__truediv__.return_value
            mock_file.exists.return_value = False
            result = get_app_version()
            self.assertEqual(result, "0.0.0")

    def test_fallback_when_file_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            tmp_path = Path(f.name)

        with patch("sniper.version.Path") as mock_path_cls:
            mock_file = mock_path_cls.return_value.resolve.return_value.parent.parent.__truediv__.return_value
            mock_file.exists.return_value = True
            mock_file.read_text.return_value = "   "
            result = get_app_version()
            self.assertEqual(result, "0.0.0")

        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
