"""Tests for the HTTP app: health check and the Telegram webhook's secret check."""

from collections.abc import AsyncIterator

import httpx
import pytest
from arq.connections import ArqRedis
from pydantic import SecretStr

from scambot.api.intake import Intake
from scambot.api.main import create_app
from scambot.api.webhooks import TELEGRAM_AUTH_HEADER
from scambot.core.locales import Locales
from scambot.core.redis import claim_update
from scambot.core.settings import Settings
from tests.fakes import FakeAdapter, fake_update

SECRET = "s3cret-token_value"


async def client_for(settings: Settings, intake: Intake) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(settings, telegram_intake=intake))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def client(
    adapter: FakeAdapter, redis: ArqRedis, locales: Locales
) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(_env_file=None, telegram_webhook_secret=SecretStr(SECRET))
    async with await client_for(settings, Intake(adapter, redis, locales)) as client:
        yield client


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_webhook_with_valid_secret_is_handled(
    client: httpx.AsyncClient, adapter: FakeAdapter
) -> None:
    response = await client.post(
        "/webhook/telegram", json=fake_update(1, "/start"), headers={TELEGRAM_AUTH_HEADER: SECRET}
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(adapter.log) == 1


async def test_webhook_asks_for_a_redelivery_while_the_update_is_in_progress(
    client: httpx.AsyncClient, adapter: FakeAdapter, redis: ArqRedis
) -> None:
    await claim_update(redis, "telegram:1")

    response = await client.post(
        "/webhook/telegram", json=fake_update(1, "/start"), headers={TELEGRAM_AUTH_HEADER: SECRET}
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert adapter.log == []


async def test_webhook_acknowledges_a_duplicate(
    client: httpx.AsyncClient, adapter: FakeAdapter
) -> None:
    for _ in range(2):
        response = await client.post(
            "/webhook/telegram",
            json=fake_update(1, "/start"),
            headers={TELEGRAM_AUTH_HEADER: SECRET},
        )
        assert response.status_code == 200

    assert len(adapter.log) == 1


@pytest.mark.parametrize(
    "headers", [{}, {TELEGRAM_AUTH_HEADER: "wrong"}, {TELEGRAM_AUTH_HEADER: ""}]
)
async def test_webhook_without_valid_secret_is_rejected(
    client: httpx.AsyncClient, adapter: FakeAdapter, headers: dict[str, str]
) -> None:
    response = await client.post(
        "/webhook/telegram", json=fake_update(1, "/start"), headers=headers
    )

    assert response.status_code == 403
    assert adapter.log == []


@pytest.mark.parametrize("value", [SECRET.upper().encode(), b"\xe9\xff", SECRET.encode() + b"x"])
async def test_webhook_rejects_near_miss_and_non_ascii_secrets(
    client: httpx.AsyncClient, adapter: FakeAdapter, value: bytes
) -> None:
    response = await client.post(
        "/webhook/telegram",
        json=fake_update(1, "/start"),
        headers={TELEGRAM_AUTH_HEADER.encode(): value},
    )

    assert response.status_code == 403
    assert adapter.log == []


async def test_webhook_is_closed_when_no_secret_is_configured(
    adapter: FakeAdapter, redis: ArqRedis, locales: Locales
) -> None:
    settings = Settings(_env_file=None)
    async with await client_for(settings, Intake(adapter, redis, locales)) as client:
        response = await client.post(
            "/webhook/telegram", json=fake_update(1, "/start"), headers={TELEGRAM_AUTH_HEADER: ""}
        )

    assert response.status_code == 403


@pytest.mark.parametrize("body", [b"[1, 2]", b"not json", b"\xff"])
async def test_webhook_rejects_a_body_that_is_not_a_json_object(
    client: httpx.AsyncClient, body: bytes
) -> None:
    response = await client.post(
        "/webhook/telegram", content=body, headers={TELEGRAM_AUTH_HEADER: SECRET}
    )

    assert response.status_code == 400


async def test_secret_is_checked_before_the_body(client: httpx.AsyncClient) -> None:
    response = await client.post("/webhook/telegram", content=b"not json")

    assert response.status_code == 403


async def test_api_docs_are_not_exposed(client: httpx.AsyncClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert (await client.get(path)).status_code == 404
