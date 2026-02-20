#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*[\"'][^\"'\n]{8,}[\"']"),
    re.compile(r"-----BEGIN (RSA|EC|OPENSSH|DSA)? ?PRIVATE KEY-----"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
]

ALLOWLIST_PATHS = {
    ".streamlit/secrets.toml.example",
    "README.md",
    "NEWS_SETUP.md",
}

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".lock", ".db", ".pyc", ".pyo", ".zip", ".tar", ".gz"
}


def get_staged_files() -> list[str]:
    out = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True)
    files = [line.strip() for line in out.splitlines() if line.strip()]
    return files


def get_tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    files = [line.strip() for line in out.splitlines() if line.strip()]
    return files


def scan_file(path: Path) -> list[str]:
    issues = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return issues

    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        for pattern in PATTERNS:
            if pattern.search(line):
                issues.append(f"{path}:{idx}: possible secret pattern")
                break
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight secret scanner for staged files")
    parser.add_argument("--staged", action="store_true", help="scan only staged files")
    args = parser.parse_args()

    if args.staged:
        try:
            files = get_staged_files()
        except Exception as exc:
            print(f"secret-scan: unable to read staged files: {exc}")
            return 1
    else:
        try:
            files = get_tracked_files()
        except Exception:
            files = [str(p) for p in Path(".").rglob("*") if p.is_file()]

    issues = []
    for raw in files:
        rel = raw.replace("\\", "/")
        if rel in ALLOWLIST_PATHS:
            continue
        path = Path(rel)
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() in SKIP_EXTENSIONS:
            continue
        if ".git/" in rel or "__pycache__/" in rel or ".venv/" in rel:
            continue
        issues.extend(scan_file(path))

    if issues:
        print("\nsecret-scan: potential secrets detected:\n")
        for issue in issues[:120]:
            print(f"- {issue}")
        print("\nCommit blocked. Remove or move secrets to .streamlit/secrets.toml (gitignored).")
        return 1

    print("secret-scan: no obvious secrets found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
