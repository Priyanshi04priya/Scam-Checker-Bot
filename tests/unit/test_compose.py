"""Tests for the container setup files (Docker is not run here)."""

import ipaddress
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scambot.core.settings import PROJECT_ROOT


def compose(name: str = "docker-compose.yml") -> dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _networks(service: dict[str, Any]) -> set[str]:
    networks = service.get("networks", [])
    return set(networks)


# --- development file --------------------------------------------------------------------


def test_dev_compose_services() -> None:
    config = compose("docker-compose.dev.yml")
    services = config["services"]

    assert config["name"] == "scambot-dev"
    assert {"postgres", "redis", "mcp"} <= set(services)
    assert services["postgres"]["ports"] == ["127.0.0.1:5432:5432"]
    assert services["redis"]["ports"] == ["127.0.0.1:6379:6379"]
    mcp = services["mcp"]
    assert mcp["command"] == ["python", "-m", "scambot.mcp_server.server"]
    assert mcp["environment"]["MCP_HOST"] == "0.0.0.0"  # noqa: S104 - inside the container
    assert "mcp:*" in mcp["environment"]["MCP_ALLOWED_HOSTS"]
    assert mcp["ports"] == ["127.0.0.1:8001:8001"]
    assert "@postgres:5432/" in mcp["environment"]["DATABASE_URL"]
    assert mcp["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "models:/models" in mcp["volumes"]
    assert "models" in config["volumes"]
    assert "media:/var/lib/scambot/media" in mcp["volumes"]
    assert "media" in config["volumes"]


# --- production file ---------------------------------------------------------------------


def test_prod_compose_has_exactly_the_expected_services() -> None:
    assert set(compose()["services"]) == {
        "caddy",
        "api",
        "worker",
        "mcp",
        "migrate",
        "postgres",
        "redis",
        "backup",
    }


def test_only_caddy_publishes_ports() -> None:
    services = compose()["services"]

    assert services["caddy"]["ports"] == ["80:80", "443:443", "443:443/udp"]
    for name, service in services.items():
        if name != "caddy":
            assert "ports" not in service, name
            assert "expose" not in service, name
        assert service.get("network_mode") != "host", name
        assert not service.get("privileged"), name


def test_networks_separate_the_edge_from_the_data_services() -> None:
    config = compose()
    services = config["services"]

    assert config["networks"]["backend"]["internal"] is True
    assert not config["networks"]["edge"].get("internal")
    on_edge = {name for name, service in services.items() if "edge" in _networks(service)}
    assert on_edge == {"caddy", "api"}
    assert _networks(services["caddy"]) == {"edge"}
    assert "edge" not in _networks(services["mcp"])
    for name in ("postgres", "redis", "migrate", "backup"):
        assert _networks(services[name]) == {"backend"}, name
    assert _networks(services["mcp"]) == {"backend", "egress"}
    assert _networks(services["worker"]) == {"backend", "egress"}


def test_data_services() -> None:
    services = compose()["services"]

    assert services["postgres"]["image"] == "pgvector/pgvector:pg17"
    redis_command = " ".join(services["redis"]["command"])
    assert "--appendonly yes" in redis_command
    # compose escapes the shell variable as $$ so that sh, not compose, expands it
    assert '--requirepass "$$REDIS_PASSWORD"' in redis_command
    assert "healthcheck" in services["postgres"]
    assert "healthcheck" in services["redis"]


def test_restart_limits_logging_and_healthchecks() -> None:
    services = compose()["services"]

    for name, service in services.items():
        assert service["restart"] == ("no" if name == "migrate" else "unless-stopped"), name
        assert "mem_limit" in service, name
        assert service["logging"]["driver"] == "json-file", name
    for name in ("postgres", "redis", "mcp", "api", "worker", "caddy"):
        assert "healthcheck" in services[name], name
    for name in ("api", "worker", "mcp", "migrate"):
        assert services[name]["cap_drop"] == ["ALL"], name
        assert "no-new-privileges:true" in services[name]["security_opt"], name


def test_start_order() -> None:
    services = compose()["services"]

    for name in ("api", "worker", "mcp"):
        assert services[name]["depends_on"]["migrate"]["condition"] == (
            "service_completed_successfully"
        )
    assert services["migrate"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["worker"]["depends_on"]["mcp"]["condition"] == "service_healthy"
    assert services["caddy"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_volumes() -> None:
    config = compose()
    services = config["services"]

    for name in ("worker", "mcp"):
        assert "models:/models" in services[name]["volumes"]
        assert "media:/var/lib/scambot/media" in services[name]["volumes"]
    assert "volumes" not in services["api"]
    assert not any("media" in volume for volume in services["caddy"]["volumes"])
    used = {
        volume.split(":")[0]
        for service in services.values()
        for volume in service.get("volumes", [])
        if not volume.startswith(".")
    }
    assert used == set(config["volumes"])
    assert set(config["volumes"]) == {
        "postgres-data",
        "redis-data",
        "models",
        "media",
        "backups",
        "caddy-data",
        "caddy-config",
    }


def test_api_trusts_only_the_proxy_address() -> None:
    config = compose()
    command = config["services"]["api"]["command"]
    caddy_address = config["services"]["caddy"]["networks"]["edge"]["ipv4_address"]
    subnet = config["networks"]["edge"]["ipam"]["config"][0]["subnet"]

    assert "--proxy-headers" in command
    assert command[command.index("--forwarded-allow-ips") + 1] == caddy_address
    assert ipaddress.ip_address(caddy_address) in ipaddress.ip_network(subnet)


def test_shared_environment_and_secrets() -> None:
    text = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    environment = compose()["x-app"]["environment"]

    assert environment["ENVIRONMENT"] == "prod"
    assert "@postgres:5432/" in environment["DATABASE_URL"]
    assert "@redis:6379" in environment["REDIS_URL"]
    assert "REDIS_PASSWORD:?" in text
    assert environment["MCP_URL"] == "http://mcp:8001/mcp"
    assert "POSTGRES_PASSWORD:?" in text
    assert "${POSTGRES_PASSWORD:-" not in text
    assert "scambot:scambot" not in text
    assert "required: false" not in text


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI is not installed")
def test_resolved_config_only_publishes_caddy(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TELEGRAM_BOT_TOKEN=dummy\n", encoding="utf-8")
    env_file = tmp_path / "vars.env"
    env_file.write_text(
        "DOMAIN=bot.example.com\nACME_EMAIL=ops@example.com\nPOSTGRES_PASSWORD=ci-pass\nREDIS_PASSWORD=ci-redis\n",
        encoding="utf-8",
    )
    base = ["docker", "compose", "--project-directory", str(tmp_path), "--env-file", str(env_file)]
    result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
        [*base, "-f", str(PROJECT_ROOT / "docker-compose.yml"), "config", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    resolved = json.loads(result.stdout)["services"]
    assert {name for name, service in resolved.items() if service.get("ports")} == {"caddy"}
    assert result.stdout.count("${") == result.stdout.count("$${")  # only escaped ones remain
    assert "ci-pass@postgres:5432" in resolved["api"]["environment"]["DATABASE_URL"]
    subprocess.run(  # noqa: S603
        ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.dev.yml"), "config", "-q"],  # noqa: S607
        capture_output=True,
        timeout=30,
        check=True,
    )
