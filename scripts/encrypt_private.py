#!/usr/bin/env python3
"""
encrypt_private.py — Encrypt command_center/private.json with a PIN.

Produces command_center/private.enc.json (safe to commit to repo).
The browser decrypts it client-side using the same PIN via Web Crypto API.

Usage:
    python scripts/encrypt_private.py
    python scripts/encrypt_private.py --pin 1234   # (PIN in args — avoid on shared machines)

Requirements:
    pip install cryptography
"""

import argparse
import base64
import getpass
import json
import os
import sys

try:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    print("ERROR: 'cryptography' package not installed. Run: pip install cryptography")
    sys.exit(1)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
INPUT_FILE   = os.path.join(REPO_ROOT, "command_center", "private.json")
OUTPUT_FILE  = os.path.join(REPO_ROOT, "command_center", "private.enc.json")

PBKDF2_ITERS = 100_000


def derive_key(pin: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERS,
    )
    return kdf.derive(pin.encode("utf-8"))


def encrypt(plaintext: bytes, pin: str) -> dict:
    salt = os.urandom(16)
    iv   = os.urandom(12)
    key  = derive_key(pin, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(iv, plaintext, None)  # no AAD
    return {
        "_note": "AES-256-GCM / PBKDF2-SHA256 / 100k iters — decrypt with PIN in browser",
        "salt":       base64.b64encode(salt).decode(),
        "iv":         base64.b64encode(iv).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
    }


def main():
    parser = argparse.ArgumentParser(description="Encrypt private.json with a PIN")
    parser.add_argument("--pin", help="PIN (omit to be prompted securely)")
    args = parser.parse_args()

    # Load plaintext
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        print("Create command_center/private.json with your private data first.")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        plaintext = f.read().encode("utf-8")

    # Validate JSON
    try:
        json.loads(plaintext)
    except json.JSONDecodeError as e:
        print(f"ERROR: private.json is not valid JSON: {e}")
        sys.exit(1)

    # Get PIN
    if args.pin:
        pin = args.pin
    else:
        pin = getpass.getpass("Enter PIN: ")
        pin2 = getpass.getpass("Confirm PIN: ")
        if pin != pin2:
            print("ERROR: PINs do not match.")
            sys.exit(1)

    if not pin:
        print("ERROR: PIN cannot be empty.")
        sys.exit(1)

    # Encrypt
    result = encrypt(plaintext, pin)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"✓ Encrypted → {OUTPUT_FILE}")
    print(f"  Salt:       {result['salt'][:16]}...")
    print(f"  IV:         {result['iv']}")
    print(f"  Ciphertext: {result['ciphertext'][:32]}... ({len(result['ciphertext'])} chars)")
    print()
    print("Next steps:")
    print("  1. git add command_center/private.enc.json")
    print("  2. git commit -m 'chore: update encrypted private data'")
    print("  3. git push")


if __name__ == "__main__":
    main()
