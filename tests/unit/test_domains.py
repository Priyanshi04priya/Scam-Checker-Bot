"""Tests for host helpers and link inspection."""

import pytest

from scambot.core.config import AppConfig
from scambot.core.domains import domain_label, has_public_suffix, registrable_domain, url_host
from scambot.mcp_server.checkers.domains import UrlFinding, inspect_url, skeleton


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("bit.ly/3x", "bit.ly"),
        ("HTTPS://WWW.SBI.CO.IN/kyc", "www.sbi.co.in"),
        ("ftp://files.example.com", None),
        ("https://", None),
        ("http://[::1", None),
    ],
)
def test_url_host(url: str, host: str | None) -> None:
    assert url_host(url) == host


def test_registrable_domain_and_label() -> None:
    assert registrable_domain("a.b.sbi.co.in") == "sbi.co.in"
    assert domain_label("netbanking.hdfcbank.com") == "hdfcbank"
    assert registrable_domain("10.0.0.1") == ""
    assert domain_label("10.0.0.1") == ""
    assert has_public_suffix("pmkisan-yojana.online")
    assert not has_public_suffix("file.txt")
    assert not has_public_suffix("127.0.0.1")


def test_skeleton_maps_lookalikes() -> None:
    assert skeleton("paytrn") == "paytm"
    assert skeleton("AMAZ0N") == "amazon"
    assert skeleton("\u0430mazon") == "amazon"  # Cyrillic a


def inspect(app_config: AppConfig, url: str) -> UrlFinding:
    patterns = app_config.scam_patterns
    return inspect_url(
        url, patterns.brands, frozenset(patterns.shorteners), app_config.thresholds.scam.fuzzy
    )


@pytest.mark.parametrize(
    ("url", "flags", "brand"),
    [
        ("sbi-kyc-update.co.in", ["brand_in_domain:sbi"], "sbi"),
        ("https://onlinesbi.sbi/login", [], None),
        ("https://www.hdfcbank.com", [], None),
        ("https://pay.google.com/x", [], None),
        ("paytrn.com", ["lookalike:paytm"], "paytm"),
        ("amaz0n-deals.com", ["lookalike:amazon"], "amazon"),
        ("www.amazon.in.verify-account.xyz", ["brand_in_subdomain:amazon"], "amazon"),
        ("flipkart-sale.in", ["brand_in_domain:flipkart"], "flipkart"),
        ("xn--pypal-4ve.com", ["punycode", "homoglyph"], None),
        ("bit.ly/3xYz", ["shortener"], None),
        ("join-now.com", [], None),
        ("example.com", [], None),
    ],
)
def test_inspect_url(app_config: AppConfig, url: str, flags: list[str], brand: str | None) -> None:
    finding = inspect(app_config, url)

    assert finding.flags == flags
    assert finding.brand == brand


def test_devanagari_idn_is_not_a_lookalike(app_config: AppConfig) -> None:
    finding = inspect(app_config, "http://xn--h2brj9c.xn--h2brj9c/")  # भारत.भारत

    assert finding.flags == ["punycode"]
    assert not finding.is_lookalike


def test_invalid_idn_is_flagged(app_config: AppConfig) -> None:
    finding = inspect(app_config, "xn--zz-.com")

    assert "invalid_idn" in finding.flags
    assert finding.is_lookalike


def test_signal_properties(app_config: AppConfig) -> None:
    assert inspect(app_config, "bit.ly/x").is_shortener
    assert not inspect(app_config, "bit.ly/x").is_lookalike
    assert inspect(app_config, "paytrn.com").is_lookalike
