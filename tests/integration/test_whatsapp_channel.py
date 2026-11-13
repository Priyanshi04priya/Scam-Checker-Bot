"""WhatsApp end to end: webhook adapter, queue, worker and the Graph API mocked with respx."""

import json
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx
from arq.connections import ArqRedis

from scambot.api.intake import Intake, Outcome
from scambot.channels.whatsapp import WhatsAppAdapter
from scambot.core.locales import Locales
from scambot.core.logging import configure_logging
from tests.conftest import PipelineDepsFactory
from tests.fakes import FakeAdapter, InMemoryStore, fake_media_update
from tests.integration.test_walking_skeleton import _skip_redis_info, run_worker  # noqa: F401
from tests.media import exif_jpeg
from tests.whatsapp import CDN, GRAPH, PHONE_ID, message, raw, settings, text

MESSAGES = f"{GRAPH}/{PHONE_ID}/messages"
SCAM = "Your electricity will be disconnected tonight. Call 9876543210 immediately to pay the bill"
IMAGE = exif_jpeg()
SLASH = re.compile(r"(?<![\w/])/(start|help|language|history|privacy|delete)\b")


@pytest.fixture
def graph() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{GRAPH}/555").respond(
            200, json={"url": CDN, "mime_type": "image/jpeg", "file_size": len(IMAGE)}
        )
        mock.get(CDN).respond(200, content=IMAGE)
        yield mock


@pytest.fixture
async def whatsapp(locales: Locales, graph: respx.MockRouter) -> AsyncIterator[WhatsAppAdapter]:
    config = settings()
    async with httpx.AsyncClient() as client:
        yield WhatsAppAdapter(
            client,
            token=config.whatsapp_token,
            phone_number_id=config.whatsapp_phone_number_id,
            api_version=config.whatsapp_api_version,
            graph_url=config.whatsapp_graph_url,
            allowlist=config.whatsapp_allowlist,
            user_id_pepper=config.user_id_pepper,
            mark_read=False,
            locales=locales,
        )


@pytest.fixture
def outbox(graph: respx.MockRouter) -> respx.Route:
    return graph.post(MESSAGES).mock(
        side_effect=lambda _r: httpx.Response(200, json={"messages": [{"id": "wamid.OUT"}]})
    )


def sent(route: respx.Route) -> list[dict[str, Any]]:
    return [json.loads(call.request.content) for call in route.calls]


def body_of(payload: dict[str, Any]) -> str:
    if payload["type"] == "interactive":
        return str(payload["interactive"]["body"]["text"])
    return str(payload["text"]["body"])


async def deliver(
    redis: ArqRedis, whatsapp: WhatsAppAdapter, locales: Locales, msg: dict[str, Any]
) -> Outcome:
    return await Intake(whatsapp, redis, locales).handle(raw(msg))


async def test_text_gets_an_ack_then_a_separate_verdict_with_buttons(
    redis: ArqRedis,
    whatsapp: WhatsAppAdapter,
    outbox: respx.Route,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
) -> None:
    assert await deliver(redis, whatsapp, locales, text(SCAM)) is Outcome.ENQUEUED
    assert await run_worker(redis, pipeline_deps(adapters={"whatsapp": whatsapp})) == 1

    ack, verdict = sent(outbox)
    en = locales.get("en")
    assert body_of(ack) == en.checking
    assert verdict["type"] == "interactive"
    assert verdict["context"] == {"message_id": "wamid.A1"}
    assert en.verdict.bands.high in body_of(verdict)
    (row,) = store.requests.values()
    assert row.status == "replied"
    buttons = verdict["interactive"]["action"]["buttons"]
    assert [b["reply"]["id"] for b in buttons] == [f"{p}:{row.id}" for p in "dhw"]
    assert len(sent(outbox)) == 2  # no edit exists: two messages in total


async def test_a_chat_style_prompt_gets_the_standard_verdict_not_an_answer(
    redis: ArqRedis,
    whatsapp: WhatsAppAdapter,
    outbox: respx.Route,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
) -> None:
    prompt = text("Please write me a poem about the capital of France and its history")

    await deliver(redis, whatsapp, locales, prompt)
    await run_worker(redis, pipeline_deps(adapters={"whatsapp": whatsapp}))

    reply = body_of(sent(outbox)[-1])
    en = locales.get("en")
    assert reply.splitlines()[0].startswith(("🟢", "🟡", "🔴", "⚪"))
    assert reply.endswith(en.disclaimer)
    assert "poem" not in reply.lower()


async def test_an_image_gets_the_same_verdict_on_both_channels(
    redis: ArqRedis,
    whatsapp: WhatsAppAdapter,
    outbox: respx.Route,
    locales: Locales,
    adapter: FakeAdapter,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
) -> None:
    adapter.files["img"] = IMAGE
    telegram = Intake(adapter, redis, locales)
    image = message("image", {"id": "555", "mime_type": "image/jpeg", "caption": "flood today"})

    assert (
        await telegram.handle(fake_media_update(1, "image", "img", caption="flood today"))
        is Outcome.ENQUEUED
    )
    assert await Intake(whatsapp, redis, locales).handle(raw(image)) is Outcome.ENQUEUED
    deps = pipeline_deps(adapters={"telegram": adapter, "whatsapp": whatsapp})
    assert await run_worker(redis, deps) == 2

    telegram_reply = adapter.log[-1][2]
    assert SLASH.sub(r"\1", telegram_reply) == body_of(sent(outbox)[-1])
    rows = sorted(store.requests.values(), key=lambda r: r.request.channel)
    assert [r.request.channel for r in rows] == ["telegram", "whatsapp"]
    assert rows[0].verdict is not None
    assert rows[1].verdict is not None
    assert (rows[0].verdict.label, rows[0].verdict.band) == (
        rows[1].verdict.label,
        rows[1].verdict.band,
    )


@pytest.mark.parametrize("code", [131047, 131030])
async def test_a_reply_outside_the_window_is_dropped_without_a_retry(
    redis: ArqRedis,
    whatsapp: WhatsAppAdapter,
    outbox: respx.Route,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
    capsys: pytest.CaptureFixture[str],
    code: int,
) -> None:
    configure_logging("INFO")
    outbox.mock(
        side_effect=[
            httpx.Response(200, json={"messages": [{"id": "wamid.ACK"}]}),
            httpx.Response(400, json={"error": {"code": code, "type": "OAuthException"}}),
        ]
    )

    await deliver(redis, whatsapp, locales, text(SCAM))
    assert await run_worker(redis, pipeline_deps(adapters={"whatsapp": whatsapp})) == 1

    (row,) = store.requests.values()
    assert row.status == "replied"
    assert len(outbox.calls) == 2
    assert "whatsapp_send_dropped" in capsys.readouterr().out


async def test_unregistered_number_leaves_no_trace(
    redis: ArqRedis,
    whatsapp: WhatsAppAdapter,
    outbox: respx.Route,
    locales: Locales,
    store: InMemoryStore,
) -> None:
    outcome = await deliver(redis, whatsapp, locales, text(SCAM, sender="919800000009"))

    assert outcome is Outcome.IGNORED
    assert not outbox.called
    assert await redis.queued_jobs() == []
    assert store.requests == {}
