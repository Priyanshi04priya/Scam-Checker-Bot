"""Tests that every package module imports and the dependency rule between packages holds."""

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import scambot

PACKAGE_ROOT = Path(scambot.__file__).parent

EXPECTED_PACKAGES = {
    "scambot.core",
    "scambot.core.llm",
    "scambot.channels",
    "scambot.api",
    "scambot.api.admin",
    "scambot.worker",
    "scambot.preprocess",
    "scambot.agent",
    "scambot.mcp_server",
    "scambot.mcp_server.checkers",
    "scambot.mcp_server.clients",
    "scambot.factcheck_index",
}


def _all_modules() -> list[str]:
    return sorted(info.name for info in pkgutil.walk_packages(scambot.__path__, prefix="scambot."))


def test_expected_packages_exist() -> None:
    assert set(_all_modules()) >= EXPECTED_PACKAGES


@pytest.mark.parametrize("module_name", _all_modules())
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def _imported_scambot_packages(source: str, package: str) -> set[str]:
    """Top-level ``scambot`` packages imported by ``source``, a module inside ``package``."""
    package_parts = package.split(".")
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # Resolve relative imports against the module's package; keep imported names so
            # ``from scambot import agent`` counts as importing ``agent``.
            base = package_parts[: len(package_parts) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            names = [f"{module}.{alias.name}" for alias in node.names]
        for name in names:
            parts = name.split(".")
            if parts[0] == "scambot" and len(parts) > 1:
                found.add(parts[1])
    return found


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import scambot.agent.verdict\n", {"agent"}),
        ("from scambot.core.settings import Settings\n", {"core"}),
        ("from scambot import agent\n", {"agent"}),
        ("from ..agent import verdict\n", {"agent"}),
        ("from .. import agent\n", {"agent"}),
        ("from . import base\n", {"channels"}),
        ("import structlog\nfrom pydantic import BaseModel\n", set()),
    ],
)
def test_imported_scambot_packages(source: str, expected: set[str]) -> None:
    assert _imported_scambot_packages(source, "scambot.channels") == expected


@pytest.mark.parametrize(
    ("package", "allowed"),
    [
        ("core", {"core"}),
        ("channels", {"core", "channels"}),
        ("preprocess", {"core", "preprocess"}),
        ("agent", {"core", "agent"}),
        ("mcp_server", {"core", "mcp_server"}),
    ],
)
def test_dependency_rule(package: str, allowed: set[str]) -> None:
    for path in (PACKAGE_ROOT / package).rglob("*.py"):
        module_package = ".".join(path.parent.relative_to(PACKAGE_ROOT.parent).parts)
        imported = _imported_scambot_packages(path.read_text(encoding="utf-8"), module_package)
        assert imported <= allowed, f"{path.relative_to(PACKAGE_ROOT)} imports {imported - allowed}"
