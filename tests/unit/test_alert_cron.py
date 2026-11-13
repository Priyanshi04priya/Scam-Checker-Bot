"""The checker error alert."""

import datetime as dt
import uuid
from collections.abc import Sequence

import pytest
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.api.admin.queries import StatsStore
from scambot.channels.base import Button
from scambot.core.config import AppConfig
from scambot.core.models import CheckerRun, Request, User
from scambot.worker.cron import ALERT_CRON, ALERT_KEY, AlertDeps, checker_error_alert
from tests.fakes import FakeAdapter

Sessions = async_sessionmaker[AsyncSession]


async def seed(
    sessions: Sessions,
    checker: str,
    total: int,
    errors: int,
    *,
    age_min: float = 1,
    code: str = "timeout",
) -> None:
    created = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=age_min)
    async with sessions() as session, session.begin():
        if await session.get(User, "u1") is None:
            session.add(User(user_hash="u1"))
            await session.flush()
        for i in range(total):
            request = Request(
                id=uuid.uuid4(),
                job_id=f"{checker}-{age_min}-{i}-{uuid.uuid4()}",
                user_hash="u1",
                channel="telegram",
                kind="text",
                claims=[],
                status="replied",
                created_at=created,
            )
            session.add(request)
            await session.flush()
            failed = i < errors
            session.add(
                CheckerRun(
                    request_id=request.id,
                    checker=checker,
                    input_hash="h",
                    status="error" if failed else "ok",
                    signal="none",
                    confidence=0.0,
                    evidence=[],
                    details={},
                    duration_ms=1,
                    error=code if failed else None,
                )
            )


@pytest.fixture
def deps_factory(
    sessions: Sessions, redis: ArqRedis, adapter: FakeAdapter, app_config: AppConfig
) -> object:
    def build(chat_id: str = "999") -> dict[str, AlertDeps]:
        return {
            ALERT_KEY: AlertDeps(
                StatsStore(sessions), redis, adapter, chat_id, app_config.thresholds.alerts
            )
        }

    return build


@pytest.mark.parametrize(
    ("total", "errors", "alerts"),
    [(20, 5, True), (20, 4, False), (4, 4, False), (5, 2, True), (5, 1, False)],
)
async def test_threshold(
    sessions: Sessions,
    adapter: FakeAdapter,
    deps_factory: object,
    total: int,
    errors: int,
    alerts: bool,
) -> None:
    await seed(sessions, "detect_deepfake", total, errors)

    sent = await checker_error_alert(deps_factory())  # type: ignore[operator]

    assert sent == (["detect_deepfake"] if alerts else [])
    assert len(adapter.log) == int(alerts)


async def test_message_is_fixed_text_without_user_data(
    sessions: Sessions, adapter: FakeAdapter, deps_factory: object
) -> None:
    await seed(sessions, "detect_deepfake", 20, 5, code="no_score")

    await checker_error_alert(deps_factory())  # type: ignore[operator]

    (_, chat, text) = adapter.log[0]
    assert chat == "999"
    assert text == (
        "Alert: detect_deepfake failed 5 of 20 times in the last 15 minutes (25%). "
        "Top errors: no_score."
    )
    assert "u1" not in text


async def test_alert_is_sent_once_an_hour(
    sessions: Sessions, adapter: FakeAdapter, redis: ArqRedis, deps_factory: object
) -> None:
    await seed(sessions, "detect_deepfake", 10, 5)
    ctx = deps_factory()  # type: ignore[operator]

    assert await checker_error_alert(ctx) == ["detect_deepfake"]
    assert await checker_error_alert(ctx) == []
    assert len(adapter.log) == 1
    assert 0 < await redis.ttl("alert:detect_deepfake") <= 3600

    await redis.delete("alert:detect_deepfake")
    assert await checker_error_alert(ctx) == ["detect_deepfake"]


async def test_failed_send_frees_the_key(
    sessions: Sessions, redis: ArqRedis, deps_factory: object
) -> None:
    class Down(FakeAdapter):
        async def send_text(
            self, chat_id: str, text: str, *, buttons: Sequence[Sequence[Button]] = ()
        ) -> None:
            msg = "platform down"
            raise ConnectionError(msg)

    await seed(sessions, "detect_deepfake", 10, 5)
    ctx = deps_factory()  # type: ignore[operator]
    ctx[ALERT_KEY] = AlertDeps(ctx[ALERT_KEY].stats, redis, Down(), "999", ctx[ALERT_KEY].limits)

    assert await checker_error_alert(ctx) == []
    assert await redis.exists("alert:detect_deepfake") == 0


async def test_no_chat_means_no_message(
    sessions: Sessions, adapter: FakeAdapter, deps_factory: object
) -> None:
    await seed(sessions, "detect_deepfake", 10, 10)

    assert await checker_error_alert(deps_factory("")) == []  # type: ignore[operator]
    assert adapter.log == []


async def test_old_runs_are_ignored(
    sessions: Sessions, adapter: FakeAdapter, deps_factory: object
) -> None:
    await seed(sessions, "detect_deepfake", 10, 10, age_min=30)

    assert await checker_error_alert(deps_factory()) == []  # type: ignore[operator]


def test_schedule() -> None:
    assert ALERT_CRON.name == "checker_error_alert"
    assert ALERT_CRON.minute == set(range(0, 60, 5))
