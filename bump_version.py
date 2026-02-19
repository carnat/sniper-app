from pathlib import Path
import re
import sys

VERSION_FILE = Path("VERSION")


def parse_semver(version: str):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version.strip())
    if not match:
        raise ValueError("VERSION must be in MAJOR.MINOR.PATCH format")
    return tuple(int(part) for part in match.groups())


def bump(version: str, part: str):
    major, minor, patch = parse_semver(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError("Part must be one of: major, minor, patch")


def main():
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py [major|minor|patch]")
        sys.exit(1)

    part = sys.argv[1].strip().lower()
    if not VERSION_FILE.exists():
        print("VERSION file not found")
        sys.exit(1)

    current = VERSION_FILE.read_text(encoding="utf-8").strip()
    new_version = bump(current, part)
    VERSION_FILE.write_text(f"{new_version}\n", encoding="utf-8")

    print(f"Bumped version: {current} -> {new_version}")
    print("Next: update CHANGELOG.md and commit release notes.")


if __name__ == "__main__":
    main()
