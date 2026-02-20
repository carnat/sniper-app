import tempfile
import unittest
from pathlib import Path

from scripts.secret_scan import scan_file


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


if __name__ == "__main__":
    unittest.main()
