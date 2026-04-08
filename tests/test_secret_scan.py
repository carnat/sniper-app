import subprocess
import tempfile
import unittest
import sys
from pathlib import Path

# Add parent directory to path so we can import scripts module
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock

from scripts.secret_scan import scan_file, get_staged_files, get_tracked_files, main


class TestSecretScan(unittest.TestCase):
    def test_scan_file_detects_api_key_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "sample.py"
            key_literal = "AKIA" + "A" * 16
            file_path.write_text(f'my_key = "{key_literal}"\n', encoding="utf-8")

            issues = scan_file(file_path)

            self.assertEqual(len(issues), 1)
            self.assertIn("possible secret pattern", issues[0])

    def test_scan_file_ignores_safe_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "safe.py"
            file_path.write_text('print("hello world")\n', encoding="utf-8")

            issues = scan_file(file_path)

            self.assertEqual(issues, [])


class TestGetStagedFiles(unittest.TestCase):
    @patch("scripts.secret_scan.subprocess.check_output")
    def test_returns_staged_files(self, mock_output):
        mock_output.return_value = "file1.py\nfile2.py\n"
        result = get_staged_files()
        self.assertEqual(result, ["file1.py", "file2.py"])
        mock_output.assert_called_once_with(
            ["git", "diff", "--cached", "--name-only"], text=True)

    @patch("scripts.secret_scan.subprocess.check_output")
    def test_returns_empty_for_no_staged(self, mock_output):
        mock_output.return_value = ""
        result = get_staged_files()
        self.assertEqual(result, [])

    @patch("scripts.secret_scan.subprocess.check_output")
    def test_strips_whitespace(self, mock_output):
        mock_output.return_value = "  file1.py  \n  file2.py  \n\n"
        result = get_staged_files()
        self.assertEqual(result, ["file1.py", "file2.py"])

    @patch("scripts.secret_scan.subprocess.check_output")
    def test_raises_on_git_failure(self, mock_output):
        mock_output.side_effect = subprocess.CalledProcessError(1, "git")
        with self.assertRaises(subprocess.CalledProcessError):
            get_staged_files()


class TestGetTrackedFiles(unittest.TestCase):
    @patch("scripts.secret_scan.subprocess.check_output")
    def test_returns_tracked_files(self, mock_output):
        mock_output.return_value = "README.md\nscripts/run.py\n"
        result = get_tracked_files()
        self.assertEqual(result, ["README.md", "scripts/run.py"])
        mock_output.assert_called_once_with(["git", "ls-files"], text=True)

    @patch("scripts.secret_scan.subprocess.check_output")
    def test_returns_empty(self, mock_output):
        mock_output.return_value = ""
        result = get_tracked_files()
        self.assertEqual(result, [])

    @patch("scripts.secret_scan.subprocess.check_output")
    def test_raises_on_git_failure(self, mock_output):
        mock_output.side_effect = subprocess.CalledProcessError(1, "git")
        with self.assertRaises(subprocess.CalledProcessError):
            get_tracked_files()


class TestMain(unittest.TestCase):
    @patch("scripts.secret_scan.get_tracked_files")
    @patch("scripts.secret_scan.scan_file")
    @patch("sys.argv", ["secret_scan.py"])
    def test_returns_zero_no_issues(self, mock_scan, mock_tracked):
        mock_tracked.return_value = []
        result = main()
        self.assertEqual(result, 0)

    @patch("scripts.secret_scan.get_staged_files")
    @patch("sys.argv", ["secret_scan.py", "--staged"])
    def test_staged_mode_failure_returns_one(self, mock_staged):
        mock_staged.side_effect = Exception("git error")
        result = main()
        self.assertEqual(result, 1)

    @patch("scripts.secret_scan.get_tracked_files")
    @patch("scripts.secret_scan.scan_file")
    @patch("sys.argv", ["secret_scan.py"])
    def test_returns_one_on_issues(self, mock_scan, mock_tracked):
        mock_tracked.return_value = ["bad.py"]
        mock_scan.return_value = ["bad.py:1: possible secret pattern"]
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "bad.py"
            bad_file.write_text('key = "secret123456789"', encoding="utf-8")
            mock_tracked.return_value = [str(bad_file)]
            result = main()
        self.assertEqual(result, 1)

    @patch("scripts.secret_scan.get_tracked_files")
    @patch("sys.argv", ["secret_scan.py"])
    def test_skips_allowlisted_paths(self, mock_tracked):
        mock_tracked.return_value = ["README.md"]
        result = main()
        self.assertEqual(result, 0)

    @patch("scripts.secret_scan.get_tracked_files")
    @patch("sys.argv", ["secret_scan.py"])
    def test_skips_binary_extensions(self, mock_tracked):
        mock_tracked.return_value = ["image.png", "data.db"]
        result = main()
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
