"""Worker configuration that deployments depend on."""

import yaml

from scambot.core.settings import PROJECT_ROOT
from scambot.worker.settings import WorkerSettings


def test_worker_publishes_a_heartbeat_the_container_healthcheck_can_read() -> None:
    assert 0 < WorkerSettings.health_check_interval <= 60


def test_worker_drain_time_covers_the_job_timeout() -> None:
    assert WorkerSettings.job_timeout < 110  # stop_grace_period in docker-compose.yml


def test_heartbeat_outlasts_a_missed_healthcheck_interval() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    check = config["services"]["worker"]["healthcheck"]
    check_every = int(check["interval"].rstrip("s"))

    # The heartbeat key lives interval + 1 seconds; beating at least twice per check keeps a
    # slow event loop from failing a check, and retries absorb a few more.
    assert WorkerSettings.health_check_interval * 2 <= check_every
    assert check["retries"] >= 3
