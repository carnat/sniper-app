"""
claude_guardrail.py — Pre-tool hook: Commander-in-loop enforcement.

Claude Code calls this before every Bash tool use (configured in .claude/settings.json).
Reads the tool input from stdin as JSON, checks for blocked terms, exits non-zero to abort.

This enforces the Commander-in-loop doctrine rule at execution time (L3).
The rule is also stated in CLAUDE.md (L1) and .claude/settings.json deny list (L2).
"""

import json
import sys

BLOCKLIST = [
    # Execution automation — the cardinal rule
    "broker",
    "trade",
    "order",
    "execute",
    # Secret files — never touch
    "secrets.toml",
    # Destructive filesystem ops
    "rm -rf",
    # Database destruction
    "drop table",
    "delete from",
    # COMET-02: NVDA exclusion from Arsenal code
    "NVDA",
    # Commander-in-loop: no build tooling in command_center/
    "node_modules",
    "package.json",
    "webpack",
]

# Patterns that must never appear in committed files
SENSITIVE_FILE_PATTERNS = [
    "private.json",     # gitignored — must never be committed
]


def main():
    try:
        data = json.load(sys.stdin)
        command = data.get("tool_input", {}).get("command", "").lower()
    except Exception:
        sys.exit(0)  # on parse failure, allow (don't block everything)

    for term in BLOCKLIST:
        if term.lower() in command:
            print(
                f"GUARDRAIL BLOCKED: '{term}' found in command. Commander-in-loop required.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Check for private.json being committed
    if "git add" in command or "git commit" in command:
        for pattern in SENSITIVE_FILE_PATTERNS:
            if pattern in command:
                print(
                    f"GUARDRAIL BLOCKED: '{pattern}' must not be committed (gitignored).",
                    file=sys.stderr,
                )
                sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
