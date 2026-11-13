"""Tests for the container setup files (Docker is not run here)."""

from typing import Any

import yaml

from scambot.core.settings import PROJECT_ROOT


def compose() -> dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_compose_services() -> None:
    services = compose()["services"]

    assert {"postgres", "redis", "mcp"} <= set(services)
    mcp = services["mcp"]
    assert mcp["command"] == ["python", "-m", "scambot.mcp_server.server"]
    assert mcp["environment"]["MCP_HOST"] == "0.0.0.0"  # noqa: S104 - inside the container
    assert "mcp:*" in mcp["environment"]["MCP_ALLOWED_HOSTS"]
    assert mcp["ports"] == ["127.0.0.1:8001:8001"]
    assert "@postgres:5432/" in mcp["environment"]["DATABASE_URL"]
    assert mcp["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "models:/models" in mcp["volumes"]
    assert "models" in compose()["volumes"]


def test_dockerfile_builds_the_project_as_a_non_root_user() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uv sync --locked --no-dev" in dockerfile
    assert "COPY config ./config" in dockerfile
    assert "USER scambot" in dockerfile
    assert dockerfile.count("--group embeddings") == 2
    assert dockerfile.count("--group media") == 2
    apt_line = next(line for line in dockerfile.splitlines() if "tesseract-ocr-hin" in line)
    assert "ffmpeg" in apt_line
    assert "MEDIA_DIR=" in dockerfile
    assert "OMP_THREAD_LIMIT=1" in dockerfile
    assert "ENV HF_HOME=/models/huggingface" in dockerfile
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert {".env", ".venv", "tests"} <= set(ignored)
