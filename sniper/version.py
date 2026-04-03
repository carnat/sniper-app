"""App version management."""

from pathlib import Path


def get_app_version() -> str:
    """Read app version from VERSION file using SemVer format."""
    try:
        version_file = Path(__file__).resolve().parent.parent / "VERSION"
        if version_file.exists():
            version_value = version_file.read_text(encoding="utf-8").strip()
            if version_value:
                return version_value
    except Exception:
        pass
    return "0.0.0"
