"""Static checks of the reverse-proxy configuration."""

import re

from fastapi.testclient import TestClient
from pydantic import SecretStr

from scambot.api.main import create_app
from scambot.core.settings import PROJECT_ROOT, Settings

TEXT = (PROJECT_ROOT / "Caddyfile").read_text(encoding="utf-8")


def test_site_and_certificate_settings() -> None:
    assert "{$DOMAIN} {" in TEXT
    assert "email {$ACME_EMAIL}" in TEXT
    assert "http://" not in TEXT
    for forbidden in ("auto_https off", "tls internal", "debug"):
        assert forbidden not in TEXT


def test_edge_hardening() -> None:
    assert "reverse_proxy api:8000" in TEXT
    assert re.search(r"request_body\s*\{\s*max_size", TEXT)
    for header in ("Strict-Transport-Security", "X-Content-Type-Options", "-Server"):
        assert header in TEXT


def test_metrics_hidden_and_admin_restricted() -> None:
    assert re.search(r"@metrics path /metrics.*\n\s*respond @metrics 404", TEXT)
    assert "not remote_ip {$ADMIN_ALLOW_CIDRS" in TEXT
    assert "respond @admin_denied 404" in TEXT


def test_unknown_paths_are_refused() -> None:
    assert re.search(r"handle \{\s*respond 404\s*\}", TEXT)


def test_internal_services_are_never_named() -> None:
    for word in ("mcp", "8001", "postgres", "redis"):
        assert word not in TEXT


def test_braces_are_balanced() -> None:
    cleaned = re.sub(r"\{\$[A-Z_]+(:[^}]*)?\}", "", TEXT)
    assert cleaned.count("{") == cleaned.count("}")


def test_proxied_paths_exist_in_the_app() -> None:
    match = re.search(r"@public path ([^\n]+)", TEXT)
    assert match is not None
    settings = Settings(
        _env_file=None,
        admin_password_hash=SecretStr("$argon2id$placeholder"),
        whatsapp_token=SecretStr("t"),
        whatsapp_phone_number_id="1",
        whatsapp_app_secret=SecretStr("s"),
    )
    client = TestClient(create_app(settings), follow_redirects=False, raise_server_exceptions=False)

    for path in match.group(1).split():
        # A route that exists answers with anything but 404 (the dashboard needs its login or
        # its stores, which this test does not provide).
        assert (
            client.get(path.removesuffix("/*") + ("/" if path.endswith("/*") else "")).status_code
            != 404
        ), path
