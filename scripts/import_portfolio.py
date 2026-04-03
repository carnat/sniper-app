#!/usr/bin/env python3
"""
import_portfolio.py — Import portfolio state from local config into data/portfolio.json.

Reads from data/portfolio_config.json (gitignored, Commander-filled locally).
Validates against Sniper Doctrine rules.
Outputs data/portfolio.json (gitignored, consumed by Command Center and Streamlit).

Usage:
    python scripts/import_portfolio.py --template    # generate blank config template
    python scripts/import_portfolio.py               # import and validate
    python scripts/import_portfolio.py --encrypt     # import + encrypt output

Requirements:
    pip install cryptography  (only for --encrypt)

Doctrine references:
    Rule 1  (Alpha Filter): doctrine_core Section 5 — China 20/20 check
    Rule 4  (Volume gate):  doctrine_core Section 5 — ADV warning
    DC-40   (Concentration): per-position 15% (Satellite) or 40% (Core) cap check
    DC-59   (HWM/Drawdown): Drawdown Freeze check
"""

from __future__ import annotations

import argparse
import base64
import datetime
import getpass
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_FILE = os.path.join(REPO_ROOT, "data", "portfolio_config.json")
OUTPUT_FILE = os.path.join(REPO_ROOT, "data", "portfolio.json")
OUTPUT_ENC_FILE = os.path.join(REPO_ROOT, "data", "portfolio.enc.json")

PBKDF2_ITERS = 100_000
TEMPLATE_TICKER = "EXAMPLE"

TEMPLATE: dict = {
    "meta": {
        "version": "1.0.0",
        "last_updated": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        "fx_rate": 32.68,
        "hwm_thb": 0,
        "ammo_stack": [],
    },
    "arsenal": [
        {
            "ticker": TEMPLATE_TICKER,
            "shares": 0,
            "avg_cost_usd": 0.0,
            "tier": "Core S2",
            "target_weight_pct": 8,
            "thesis_date": "2026-01-01",
            "thesis_status": "INTACT",
            "flags": [],
        }
    ],
    "vault": {
        "e_class": [],
        "locked": [],
        "krungsri": [],
    },
    "shield": [],
    "watchtower": {
        "core": [],
        "satellite": [],
    },
    "standing_orders": [],
    "blacklist": [],
}


# ---------------------------------------------------------------------------
# Encryption helpers (mirrors encrypt_private.py)
# ---------------------------------------------------------------------------


def _derive_key(pin: str, salt: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        print("ERROR: 'cryptography' package not installed. Run: pip install cryptography")
        sys.exit(1)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERS,
    )
    return kdf.derive(pin.encode("utf-8"))


def _encrypt(plaintext: bytes, pin: str) -> dict:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        print("ERROR: 'cryptography' package not installed. Run: pip install cryptography")
        sys.exit(1)

    salt = os.urandom(16)
    iv = os.urandom(12)
    key = _derive_key(pin, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(iv, plaintext, None)
    return {
        "_note": "AES-256-GCM / PBKDF2-SHA256 / 100k iters — decrypt with PIN in browser",
        "salt": base64.b64encode(salt).decode(),
        "iv": base64.b64encode(iv).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationWarning:
    def __init__(self, rule: str, message: str):
        self.rule = rule
        self.message = message

    def __str__(self) -> str:
        return f"  ⚠️  [{self.rule}] {self.message}"


class ValidationError:
    def __init__(self, rule: str, message: str):
        self.rule = rule
        self.message = message

    def __str__(self) -> str:
        return f"  ❌ [{self.rule}] {self.message}"


def _validate(config: dict) -> tuple[list[ValidationError], list[ValidationWarning]]:
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    blacklist_tickers = {entry.get("ticker", "").upper() for entry in config.get("blacklist", [])}

    # --- Rule 1: China check (blacklist) ---
    for pos in config.get("arsenal", []):
        ticker = pos.get("ticker", "").upper()
        if ticker in blacklist_tickers:
            errors.append(
                ValidationError(
                    "Rule 1 (China/Alpha Filter)",
                    f"{ticker} is in the blacklist — permanently banned from Arsenal",
                )
            )

    for wt_list in config.get("watchtower", {}).values():
        for entry in wt_list:
            ticker = entry.get("ticker", "").upper()
            if ticker in blacklist_tickers:
                errors.append(
                    ValidationError(
                        "Rule 1 (China/Alpha Filter)",
                        f"{ticker} is in the blacklist — permanently banned from Watchtower",
                    )
                )

    # --- Rule 4: Volume gate warning (ADV not available at import time) ---
    for pos in config.get("arsenal", []):
        ticker = pos.get("ticker", TEMPLATE_TICKER)
        if ticker != TEMPLATE_TICKER and pos.get("shares", 0) == 0:
            warnings.append(
                ValidationWarning(
                    "Rule 4 (Volume Gate)",
                    f"{ticker}: 0 shares held — ADV gate must be verified at execution time",
                )
            )

    # --- Concentration check ---
    fx_rate = config.get("meta", {}).get("fx_rate", 33.0)
    arsenal_positions = config.get("arsenal", [])

    # Compute rough total liquid (USD Arsenal at cost basis as proxy — no live prices at import time)
    total_liquid_usd: float = 0.0
    for pos in arsenal_positions:
        shares = pos.get("shares", 0)
        cost = pos.get("avg_cost_usd", 0)
        total_liquid_usd += shares * cost

    # Add E-Class vault at 80% haircut (in THB, convert to USD)
    for fund in config.get("vault", {}).get("e_class", []):
        value_thb = fund.get("current_value_thb", 0)
        total_liquid_usd += (value_thb * 0.80) / fx_rate

    if total_liquid_usd > 0:
        for pos in arsenal_positions:
            ticker = pos.get("ticker", "")
            shares = pos.get("shares", 0)
            cost = pos.get("avg_cost_usd", 0)
            position_value = shares * cost
            pct = (position_value / total_liquid_usd) * 100

            tier = pos.get("tier", "")
            if "satellite" in tier.lower() or ticker in {"ASTS", "ONDS"}:
                if pct > 15:
                    warnings.append(
                        ValidationWarning(
                            "DC-40 Concentration",
                            f"{ticker}: Satellite position at ~{pct:.1f}% of liquid portfolio "
                            f"(cap: 15%)",
                        )
                    )
            else:
                if pct > 40:
                    warnings.append(
                        ValidationWarning(
                            "DC-40 Concentration",
                            f"{ticker}: Core position at ~{pct:.1f}% of liquid portfolio "
                            f"(trim trigger: 40%)",
                        )
                    )

    # --- HWM / Drawdown Freeze check ---
    hwm_thb = config.get("meta", {}).get("hwm_thb", 0)
    if hwm_thb > 0:
        # Estimate current liquid in THB (rough proxy using cost basis)
        current_liquid_thb = total_liquid_usd * fx_rate

        # Add Tier 1 cash if provided
        tier1_cash_thb = config.get("meta", {}).get("tier1_cash_thb", 0)
        current_liquid_thb += tier1_cash_thb

        freeze_threshold = hwm_thb * 0.80
        amber_threshold = hwm_thb * 0.90

        if current_liquid_thb < freeze_threshold:
            errors.append(
                ValidationError(
                    "DC-59 Drawdown Freeze",
                    f"Estimated liquid ฿{current_liquid_thb:,.0f} is below Freeze threshold "
                    f"฿{freeze_threshold:,.0f} (HWM × 0.80). All DCA blocked.",
                )
            )
        elif current_liquid_thb < amber_threshold:
            warnings.append(
                ValidationWarning(
                    "DC-59 Amber Zone",
                    f"Estimated liquid ฿{current_liquid_thb:,.0f} is in Amber Zone "
                    f"(฿{freeze_threshold:,.0f} – ฿{amber_threshold:,.0f}). "
                    f"Max DCA: 50% of normal cycle.",
                )
            )

    return errors, warnings


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def _generate_template() -> None:
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)

    if os.path.exists(CONFIG_FILE):
        print(f"ERROR: {CONFIG_FILE} already exists. Remove it first to regenerate template.")
        sys.exit(1)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(TEMPLATE, f, indent=2)

    print(f"✓ Template generated → {CONFIG_FILE}")
    print()
    print("Next steps:")
    print("  1. Open data/portfolio_config.json and fill in your actual positions")
    print("  2. Run: python scripts/import_portfolio.py")
    print("  3. Optionally: python scripts/import_portfolio.py --encrypt")
    print()
    print("⚠️  data/portfolio_config.json is gitignored — never commit it.")


# ---------------------------------------------------------------------------
# Import + validate
# ---------------------------------------------------------------------------


def _import(encrypt: bool, pin: str | None) -> None:
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: {CONFIG_FILE} not found.")
        print("Generate a template first: python scripts/import_portfolio.py --template")
        sys.exit(1)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: portfolio_config.json is not valid JSON: {e}")
        sys.exit(1)

    print("Validating against Sniper Doctrine rules...")
    errors, warnings = _validate(config)

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(w)

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(e)
        print()
        print("Import blocked by validation errors. Correct the config and retry.")
        sys.exit(1)

    if not warnings and not errors:
        print("  ✓ All validation checks passed.")

    # Enrich with metadata
    output = dict(config)
    output["_generated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output["_source"] = "import_portfolio.py"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Portfolio imported → {OUTPUT_FILE}")
    print(f"  Positions: {len(config.get('arsenal', []))} Arsenal, "
          f"{len(config.get('shield', []))} Shield")
    wt = config.get("watchtower", {})
    print(f"  Watchtower: {len(wt.get('core', []))} Core, {len(wt.get('satellite', []))} Satellite")
    print(f"  Standing Orders: {len(config.get('standing_orders', []))}")

    if encrypt:
        _encrypt_output(pin)


# ---------------------------------------------------------------------------
# Encryption step
# ---------------------------------------------------------------------------


def _encrypt_output(pin: str | None) -> None:
    if not os.path.exists(OUTPUT_FILE):
        print(f"ERROR: {OUTPUT_FILE} not found. Run import first.")
        sys.exit(1)

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        plaintext = f.read().encode("utf-8")

    if pin is None:
        pin = getpass.getpass("Enter encryption PIN: ")
        pin2 = getpass.getpass("Confirm PIN: ")
        if pin != pin2:
            print("ERROR: PINs do not match.")
            sys.exit(1)

    if not pin:
        print("ERROR: PIN cannot be empty.")
        sys.exit(1)

    result = _encrypt(plaintext, pin)

    with open(OUTPUT_ENC_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"✓ Encrypted → {OUTPUT_ENC_FILE}")
    print(f"  Salt:       {result['salt'][:16]}...")
    print(f"  Ciphertext: {result['ciphertext'][:32]}... ({len(result['ciphertext'])} chars)")
    print()
    print("⚠️  data/portfolio.enc.json is gitignored by default.")
    print("   Only commit it if you intend to share encrypted portfolio data.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import portfolio from data/portfolio_config.json (gitignored) "
            "→ data/portfolio.json (gitignored)"
        )
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Generate a blank portfolio_config.json template (no real data)",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Also encrypt data/portfolio.json after import",
    )
    parser.add_argument(
        "--pin",
        help="PIN for encryption (omit to be prompted securely)",
    )
    args = parser.parse_args()

    if args.template:
        _generate_template()
    else:
        _import(encrypt=args.encrypt, pin=args.pin)


if __name__ == "__main__":
    main()
