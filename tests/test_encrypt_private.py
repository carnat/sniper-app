"""Tests for scripts/encrypt_private.py."""

import base64
import os
import sys
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import encrypt_private
from encrypt_private import derive_key, encrypt


class TestDeriveKey(unittest.TestCase):
    def test_derive_key_is_deterministic_for_same_pin_and_salt(self):
        salt = b"a" * 16

        key_one = derive_key("1234", salt)
        key_two = derive_key("1234", salt)

        self.assertEqual(key_one, key_two)
        self.assertEqual(len(key_one), 32)

    def test_derive_key_changes_when_salt_changes(self):
        key_one = derive_key("1234", b"a" * 16)
        key_two = derive_key("1234", b"b" * 16)

        self.assertNotEqual(key_one, key_two)


class TestEncrypt(unittest.TestCase):
    def test_encrypt_returns_expected_structure(self):
        result = encrypt(b'{"secret": true}', "1234")

        self.assertIn("salt", result)
        self.assertIn("iv", result)
        self.assertIn("ciphertext", result)
        self.assertIsInstance(result["salt"], str)
        self.assertIsInstance(result["iv"], str)
        self.assertIsInstance(result["ciphertext"], str)

    def test_encrypt_decrypt_roundtrip(self):
        plaintext = b'{"watchlist": ["AAPL", "MSFT"]}'
        result = encrypt(plaintext, "4321")

        salt = base64.b64decode(result["salt"])
        iv = base64.b64decode(result["iv"])
        ciphertext = base64.b64decode(result["ciphertext"])
        decrypted = AESGCM(derive_key("4321", salt)).decrypt(iv, ciphertext, None)

        self.assertEqual(decrypted, plaintext)

    def test_different_pins_produce_different_ciphertexts(self):
        plaintext = b'{"secret": "value"}'
        fixed_salt = b"s" * 16
        fixed_iv = b"i" * 12

        with patch("encrypt_private.os.urandom", side_effect=[fixed_salt, fixed_iv, fixed_salt, fixed_iv]):
            first = encrypt(plaintext, "1111")
            second = encrypt(plaintext, "2222")

        self.assertEqual(first["salt"], second["salt"])
        self.assertEqual(first["iv"], second["iv"])
        self.assertNotEqual(first["ciphertext"], second["ciphertext"])

    def test_empty_plaintext_encrypts_without_error(self):
        result = encrypt(b"", "1234")

        salt = base64.b64decode(result["salt"])
        iv = base64.b64decode(result["iv"])
        ciphertext = base64.b64decode(result["ciphertext"])
        decrypted = AESGCM(derive_key("1234", salt)).decrypt(iv, ciphertext, None)

        self.assertEqual(decrypted, b"")


if __name__ == "__main__":
    unittest.main()
