"""Tests for the scripts under deploy/ (run with shim commands, never a real database)."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from scambot.core.settings import PROJECT_ROOT

DEPLOY = PROJECT_ROOT / "deploy"
SCRIPTS = sorted(DEPLOY.glob("*.sh"))
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not installed")


def _shim(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(
    script: str, *args: str, path: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {"PATH": f"{path}:/usr/bin:/bin", "HOME": str(path), **(extra_env or {})}
    return subprocess.run(  # noqa: S603 - fixed arguments, no shell
        [BASH or "bash", str(DEPLOY / script), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )


def test_scripts_exist_and_are_well_formed() -> None:
    assert {path.name for path in SCRIPTS} == {"backup.sh", "healthcheck.sh", "restore.sh"}
    for script in SCRIPTS:
        text = script.read_text(encoding="utf-8")
        assert text.splitlines()[0] in {"#!/usr/bin/env bash", "#!/bin/bash"}
        assert "set -euo pipefail" in text
        assert os.access(script, os.X_OK), script.name
        assert "| sh" not in text
        assert "| bash" not in text
        syntax = subprocess.run(  # noqa: S603
            [BASH or "bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert syntax.returncode == 0, syntax.stderr


def test_backup_script_uses_the_expected_tools() -> None:
    text = (DEPLOY / "backup.sh").read_text(encoding="utf-8")

    for needle in ("pg_dump", "--format=custom", "pg_restore --list", "-mtime", "BACKUP_KEEP_DAYS"):
        assert needle in text
    assert "PGPASSWORD=" not in text


@pytest.fixture
def backup_env(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    shims = tmp_path / "bin"
    shims.mkdir()
    backups = tmp_path / "backups"
    backups.mkdir()
    _shim(
        shims,
        "pg_dump",
        'while [ $# -gt 0 ]; do [ "$1" = --file ] && echo data > "$2"; shift; done',
    )
    _shim(shims, "pg_restore", "exit 0")
    return shims, backups, {"BACKUP_DIR": str(backups), "BACKUP_KEEP_DAYS": "14"}


def test_backup_creates_a_private_dump_and_prunes_old_ones(
    backup_env: tuple[Path, Path, dict[str, str]],
) -> None:
    shims, backups, env = backup_env
    old = backups / "scambot-20200101T000000Z.dump"
    recent = backups / "scambot-20990101T000000Z.dump"
    old.write_text("old", encoding="utf-8")
    recent.write_text("recent", encoding="utf-8")
    thirty_days_ago = old.stat().st_mtime - 30 * 86400
    os.utime(old, (thirty_days_ago, thirty_days_ago))

    result = _run("backup.sh", path=shims, extra_env=env)

    assert result.returncode == 0, result.stderr
    assert not old.exists()
    assert recent.exists()
    created = [p for p in backups.glob("scambot-*.dump") if p not in {old, recent}]
    assert len(created) == 1
    assert stat.S_IMODE(created[0].stat().st_mode) == 0o600
    assert not list(backups.glob(".tmp-*"))


def test_failed_dump_leaves_no_file_and_keeps_old_backups(
    backup_env: tuple[Path, Path, dict[str, str]],
) -> None:
    shims, backups, env = backup_env
    _shim(shims, "pg_dump", "exit 1")
    old = backups / "scambot-20200101T000000Z.dump"
    old.write_text("old", encoding="utf-8")
    thirty_days_ago = old.stat().st_mtime - 30 * 86400
    os.utime(old, (thirty_days_ago, thirty_days_ago))

    result = _run("backup.sh", path=shims, extra_env=env)

    assert result.returncode != 0
    assert list(backups.glob("scambot-*.dump")) == [old]
    assert not list(backups.glob(".tmp-*"))


@pytest.mark.parametrize("days", ["0", "-1", "abc"])
def test_backup_rejects_a_bad_retention(
    backup_env: tuple[Path, Path, dict[str, str]], days: str
) -> None:
    shims, backups, env = backup_env

    result = _run("backup.sh", path=shims, extra_env={**env, "BACKUP_KEEP_DAYS": days})

    assert result.returncode != 0
    assert not list(backups.glob("scambot-*.dump"))


def test_backup_rejects_unknown_arguments(backup_env: tuple[Path, Path, dict[str, str]]) -> None:
    shims, _backups, env = backup_env

    assert _run("backup.sh", "--nope", path=shims, extra_env=env).returncode == 2


def test_healthcheck_needs_a_domain(tmp_path: Path) -> None:
    assert _run("healthcheck.sh", path=tmp_path).returncode == 2


@pytest.mark.parametrize(("body", "code"), [('{"status":"ok"}', 0), ('{"status":"down"}', 1)])
def test_healthcheck_reads_the_health_response(tmp_path: Path, body: str, code: int) -> None:
    _shim(tmp_path, "curl", f"echo '{body}'")

    result = _run("healthcheck.sh", "bot.example.com", path=tmp_path)

    assert result.returncode == code


def test_healthcheck_fails_when_the_endpoint_is_unreachable(tmp_path: Path) -> None:
    _shim(tmp_path, "curl", "exit 7")

    assert _run("healthcheck.sh", "bot.example.com", path=tmp_path).returncode == 1


def test_healthcheck_flags_an_unhealthy_service(tmp_path: Path) -> None:
    _shim(tmp_path, "curl", 'echo \'{"status":"ok"}\'')
    _shim(
        tmp_path,
        "docker",
        "printf 'api|healthy|running\\nworker|unhealthy|running\\nmigrate||exited\\n'",
    )
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")

    result = _run(
        "healthcheck.sh", "bot.example.com", path=tmp_path, extra_env={"COMPOSE_DIR": str(tmp_path)}
    )

    assert result.returncode == 1
    assert "worker" in result.stderr


def test_restore_without_confirmation_does_not_touch_docker(tmp_path: Path) -> None:
    calls = tmp_path / "docker-calls"
    _shim(tmp_path, "docker", f'echo "$@" >> "{calls}"')
    dump = tmp_path / "scambot.dump"
    dump.write_text("x", encoding="utf-8")

    assert _run("restore.sh", str(dump), path=tmp_path).returncode != 0
    assert _run("restore.sh", path=tmp_path).returncode != 0
    assert (
        _run("restore.sh", str(tmp_path / "missing.dump"), "--yes", path=tmp_path).returncode == 1
    )
    assert not calls.exists()


def test_restore_continues_after_pg_restore_warnings(tmp_path: Path) -> None:
    calls = tmp_path / "docker-calls"
    _shim(
        tmp_path,
        "docker",
        f'echo "$@" >> "{calls}"; if [ "$2" = exec ]; then cat >/dev/null; exit 1; fi',
    )
    dump = tmp_path / "scambot.dump"
    dump.write_text("x", encoding="utf-8")

    result = _run("restore.sh", str(dump), "--yes", path=tmp_path)

    assert result.returncode == 0
    assert "run --rm migrate" in calls.read_text(encoding="utf-8")
    assert "start mcp api worker" in calls.read_text(encoding="utf-8")
