"""Tests for content fingerprints."""

from scambot.preprocess.hashing import normalise_text, text_sha256


def test_normalise_text() -> None:
    assert normalise_text("  Your  KYC\n\tEXPIRES  ") == "your kyc expires"


def test_trivial_edits_hash_the_same() -> None:
    assert text_sha256("Your KYC expires") == text_sha256("  your kyc\nexpires ")
    assert text_sha256("Your KYC expires") != text_sha256("Your KYC expired")
    assert len(text_sha256("बिजली") or "") == 64


def test_empty_text_has_no_hash() -> None:
    assert text_sha256("") is None
    assert text_sha256(" \n ") is None
