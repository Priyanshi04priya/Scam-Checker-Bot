"""The development and production environment examples must not drift apart."""

import re

from scambot.core.settings import PROJECT_ROOT, Settings

PROVIDED_BY_COMPOSE = {
    "DATABASE_URL",
    "REDIS_URL",
    "MCP_URL",
    "MCP_HOST",
    "MCP_PORT",
    "MCP_ALLOWED_HOSTS",
    "MEDIA_DIR",
}


def _names(path: str) -> dict[str, str]:
    text = (PROJECT_ROOT / path).read_text(encoding="utf-8")
    return dict(re.findall(r"^([A-Z][A-Z0-9_]*)=(.*)$", text, re.M))


def test_every_setting_is_documented_in_the_development_example() -> None:
    documented = _names(".env.example")
    for field in Settings.model_fields:
        assert field.upper() in documented, field


def test_production_example_covers_the_development_example() -> None:
    dev = set(_names(".env.example"))
    prod = _names("deploy/prod.env.example")

    missing = dev - set(prod) - PROVIDED_BY_COMPOSE - {"POSTGRES_USER", "POSTGRES_DB"}
    assert not missing
    assert {
        "DOMAIN",
        "ACME_EMAIL",
        "ADMIN_ALLOW_CIDRS",
        "POSTGRES_PASSWORD",
        "REDIS_PASSWORD",
    } <= set(prod)
    assert not PROVIDED_BY_COMPOSE & set(prod)
    assert prod["ENVIRONMENT"] == "prod"


def test_production_example_contains_no_secret_values() -> None:
    prod = _names("deploy/prod.env.example")
    secrets = [
        name
        for name in prod
        if re.search(r"(TOKEN|SECRET|KEY|PASSWORD|PEPPER|HASH)$", name)
        and name not in {"WHATSAPP_VERIFY_TOKEN_LABEL"}
    ]

    assert secrets
    for name in secrets:
        assert prod[name] == "", name
