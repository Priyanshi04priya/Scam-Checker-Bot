"""The WhatsApp webhook routes, with the real adapter on a mocked Graph API."""

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx
from arq.connections import ArqRedis
from pydantic import SecretStr

from scambot.api.intake import Intake
from scambot.api.main import create_app
from scambot.channels.whatsapp import WhatsAppAdapter
from scambot.core.locales import Locales
from scambot.core.logging import configure_logging, get_logger
from scambot.core.redis import claim_update
from scambot.core.security import user_hash
from scambot.core.settings import DEV_USER_ID_PEPPER, Settings
from scambot.core.store import NewRequest
from tests.fakes import FakeAdapter, InMemoryStore
from tests.whatsapp import (
    GRAPH,
    OTHER_SENDER,
    PHONE_ID,
    SENDER,
    STRANGER,
    VERIFY_TOKEN,
    button,
    payload,
    settings,
    sign,
    signed,
    status_payload,
    text,
)

MESSAGES = f"{GRAPH}/{PHONE_ID}/messages"
SPAM = "Your electricity will be disconnected tonight, call 9876543210 now to pay the bill"


def ok() -> httpx.Response:
    return httpx.Response(200, json={"messages": [{"id": "wamid.OUT"}]})


def sent(route: respx.Route) -> list[dict[str, Any]]:
    return [json.loads(call.request.content) for call in route.calls]


def bodies(route: respx.Route) -> list[str]:
    return [
        p["text"]["body"] if p["type"] == "text" else p["interactive"]["body"]["text"]
        for p in sent(route)
        if "type" in p and p["type"] != "reaction" and "status" not in p
    ]


@pytest.fixture
def graph() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mock:
        mock.post(MESSAGES).mock(side_effect=lambda _request: ok())
        yield mock


@pytest.fixture
def messages(graph: respx.MockRouter) -> respx.Route:
    return graph.routes[0]


@pytest.fixture
def users() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
async def api(
    http: httpx.AsyncClient, redis: ArqRedis, locales: Locales, users: InMemoryStore, graph: Any
) -> AsyncIterator[httpx.AsyncClient]:
    config = settings()
    adapter = WhatsAppAdapter(
        http,
        token=config.whatsapp_token,
        phone_number_id=config.whatsapp_phone_number_id,
        api_version=config.whatsapp_api_version,
        graph_url=config.whatsapp_graph_url,
        allowlist=config.whatsapp_allowlist,
        user_id_pepper=config.user_id_pepper,
        mark_read=False,
        locales=locales,
    )
    intake = Intake(adapter, redis, locales, users=users)
    app = create_app(config, whatsapp_intake=intake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


async def post(api: httpx.AsyncClient, data: dict[str, Any]) -> httpx.Response:
    body, headers = signed(data)
    return await api.post("/webhook/whatsapp", content=body, headers=headers)


# --------------------------------------------------------------------- verification


async def test_handshake_echoes_the_challenge(api: httpx.AsyncClient) -> None:
    response = await api.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "12345",
        },
    )

    assert response.status_code == 200
    assert response.text == "12345"


@pytest.mark.parametrize(
    "params",
    [
        {"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "1"},
        {"hub.mode": "subscribe", "hub.challenge": "1"},
        {"hub.mode": "unsubscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "1"},
    ],
)
async def test_handshake_refuses_wrong_requests(
    api: httpx.AsyncClient, params: dict[str, str]
) -> None:
    assert (await api.get("/webhook/whatsapp", params=params)).status_code == 403


async def test_handshake_rejects_an_unsafe_challenge(api: httpx.AsyncClient) -> None:
    params = {"hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "<b>"}

    assert (await api.get("/webhook/whatsapp", params=params)).status_code == 400


async def test_empty_configured_token_never_matches(
    redis: ArqRedis, locales: Locales, http: httpx.AsyncClient
) -> None:
    app = create_app(
        settings(whatsapp_verify_token=""), whatsapp_intake=Intake(FakeAdapter(), redis, locales)
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        params = {"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "1"}
        assert (await c.get("/webhook/whatsapp", params=params)).status_code == 403


async def test_routes_are_404_when_disabled(redis: ArqRedis, locales: Locales) -> None:
    app = create_app(
        Settings(_env_file=None), whatsapp_intake=Intake(FakeAdapter(), redis, locales)
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/webhook/whatsapp")).status_code == 404
        assert (await c.post("/webhook/whatsapp", content=b"{}")).status_code == 404


async def test_the_verification_request_is_logged_without_the_token(
    api: httpx.AsyncClient, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO")
    get_logger().info(
        "request",
        url=f"/webhook/whatsapp?hub.mode=subscribe&hub.verify_token={VERIFY_TOKEN}&hub.challenge=1",
    )

    assert VERIFY_TOKEN not in capsys.readouterr().out


# ------------------------------------------------------------------------- delivery


async def test_signed_message_is_acknowledged_and_queued(
    api: httpx.AsyncClient, messages: respx.Route, redis: ArqRedis
) -> None:
    response = await post(api, payload(text(SPAM)))

    assert response.status_code == 200
    assert bodies(messages) == ["Checking…"]
    assert len(await redis.queued_jobs()) == 1


@pytest.mark.parametrize("header", [None, "sha256=" + "0" * 64, "garbage"])
async def test_bad_signature_is_refused(
    api: httpx.AsyncClient, messages: respx.Route, redis: ArqRedis, header: str | None
) -> None:
    body, _ = signed(payload(text(SPAM)))
    headers = {"X-Hub-Signature-256": header} if header else {}

    response = await api.post("/webhook/whatsapp", content=body, headers=headers)

    assert response.status_code == 403
    assert not messages.called
    assert await redis.queued_jobs() == []


async def test_signature_covers_the_raw_bytes(
    api: httpx.AsyncClient, messages: respx.Route
) -> None:
    data = payload(text(SPAM))
    original = json.dumps(data).encode()
    reserialised = json.dumps(data, indent=1).encode()

    response = await api.post(
        "/webhook/whatsapp", content=reserialised, headers={"X-Hub-Signature-256": sign(original)}
    )

    assert response.status_code == 403
    assert not messages.called


async def test_oversized_body_is_refused(api: httpx.AsyncClient) -> None:
    body = b"{" + b" " * 1_100_000 + b"}"

    response = await api.post(
        "/webhook/whatsapp", content=body, headers={"X-Hub-Signature-256": sign(body)}
    )

    assert response.status_code == 413


async def test_non_json_with_a_valid_signature_is_a_bad_request(api: httpx.AsyncClient) -> None:
    body = b"not json"

    response = await api.post(
        "/webhook/whatsapp", content=body, headers={"X-Hub-Signature-256": sign(body)}
    )

    assert response.status_code == 400


async def test_json_that_is_not_an_object_is_a_bad_request(api: httpx.AsyncClient) -> None:
    body = b"[1]"

    response = await api.post(
        "/webhook/whatsapp", content=body, headers={"X-Hub-Signature-256": sign(body)}
    )

    assert response.status_code == 400


async def test_other_objects_are_acknowledged_and_ignored(
    api: httpx.AsyncClient, messages: respx.Route
) -> None:
    assert (await post(api, {"object": "page", "entry": []})).status_code == 200
    assert not messages.called


async def test_unregistered_sender_gets_no_reply_at_all(
    api: httpx.AsyncClient, messages: respx.Route, redis: ArqRedis
) -> None:
    response = await post(api, payload(text(SPAM, sender=STRANGER)))

    assert response.status_code == 200
    assert not messages.called
    assert await redis.queued_jobs() == []


async def test_status_updates_and_other_numbers_are_ignored(
    api: httpx.AsyncClient, messages: respx.Route
) -> None:
    assert (await post(api, status_payload())).status_code == 200
    assert (await post(api, payload(text(SPAM), phone_id="999"))).status_code == 200
    assert not messages.called


async def test_a_repeated_payload_is_handled_once(
    api: httpx.AsyncClient, messages: respx.Route, redis: ArqRedis
) -> None:
    data = payload(text(SPAM))

    assert (await post(api, data)).status_code == 200
    assert (await post(api, data)).status_code == 200

    assert len(bodies(messages)) == 1
    assert len(await redis.queued_jobs()) == 1


async def test_in_progress_message_asks_for_a_redelivery(
    api: httpx.AsyncClient, messages: respx.Route, redis: ArqRedis
) -> None:
    await claim_update(redis, "whatsapp:wamid.A1")
    data = payload(text(SPAM, wamid="wamid.A1"), text("help", wamid="wamid.B2"))

    response = await post(api, data)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert len(bodies(messages)) == 1  # the other message was still handled


async def test_two_messages_in_one_payload_are_both_handled(
    api: httpx.AsyncClient, messages: respx.Route, redis: ArqRedis
) -> None:
    data = payload(text(SPAM, wamid="wamid.A1"), text(SPAM + " again", wamid="wamid.B2"))

    assert (await post(api, data)).status_code == 200

    assert bodies(messages) == ["Checking…", "Checking…"]
    assert len(await redis.queued_jobs()) == 2


# ----------------------------------------------------------------------- commands


async def test_help_reply_has_no_slash_commands(
    api: httpx.AsyncClient, messages: respx.Route, locales: Locales
) -> None:
    await post(api, payload(text("help")))

    (reply,) = bodies(messages)
    assert "/start" not in reply
    assert "/delete" not in reply
    assert "delete —" in reply


async def test_delete_erases_and_confirms(
    api: httpx.AsyncClient, messages: respx.Route, users: InMemoryStore
) -> None:
    await post(api, payload(text("delete", wamid="wamid.D1")))
    (ask,) = bodies(messages)
    assert "0 checks" not in ask
    assert "again" in ask

    await post(api, payload(text("delete", wamid="wamid.D2")))
    assert "0 checks" in bodies(messages)[1]


async def test_delete_confirmation_is_single_use(
    api: httpx.AsyncClient, messages: respx.Route
) -> None:
    for n in range(3):
        await post(api, payload(text("delete", wamid=f"wamid.D{n}")))

    replies = bodies(messages)
    assert "0 checks" in replies[1]
    assert "0 checks" not in replies[2]


async def test_language_and_history(
    api: httpx.AsyncClient, messages: respx.Route, users: InMemoryStore
) -> None:
    await post(api, payload(text("language hindi", wamid="wamid.L1")))
    hashed = user_hash(SENDER, SecretStr(DEV_USER_ID_PEPPER))
    assert await users.get_language(hashed) == "hi"

    await post(api, payload(text("history", wamid="wamid.H1")))
    assert len(bodies(messages)) == 2


# ----------------------------------------------------------------------- buttons


async def seeded_request(users: InMemoryStore, owner: str) -> uuid.UUID:
    hashed = user_hash(owner, SecretStr(DEV_USER_ID_PEPPER))
    request = NewRequest("whatsapp:seed", hashed, "whatsapp", "text", None, "en")
    return (await users.start(request)).id


async def test_helpful_press_stores_feedback_and_thanks(
    api: httpx.AsyncClient, messages: respx.Route, users: InMemoryStore, locales: Locales
) -> None:
    request_id = await seeded_request(users, SENDER)

    await post(api, payload(button(f"h:{request_id}")))

    assert len(bodies(messages)) == 1
    assert users.feedback == {(request_id, user_hash(SENDER, SecretStr(DEV_USER_ID_PEPPER))): True}
    assert bodies(messages) == [locales.get("en").feedback.thanks_helpful]


async def test_details_press_sends_the_details(
    api: httpx.AsyncClient, messages: respx.Route, users: InMemoryStore
) -> None:
    request_id = await seeded_request(users, SENDER)

    await post(api, payload(button(f"d:{request_id}")))

    assert len(bodies(messages)) == 1


async def test_someone_elses_request_is_unavailable(
    api: httpx.AsyncClient, messages: respx.Route, users: InMemoryStore, locales: Locales
) -> None:
    request_id = await seeded_request(users, OTHER_SENDER)

    await post(api, payload(button(f"h:{request_id}")))

    assert users.feedback == {}
    assert bodies(messages) == [locales.get("en").feedback.unavailable]


async def test_post_without_an_intake_is_a_503_not_a_crash(redis: ArqRedis) -> None:
    app = create_app(settings())
    app.state.whatsapp_intake = None
    body, headers = signed({"object": "whatsapp_business_account", "entry": []})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        response = await c.post("/webhook/whatsapp", content=body, headers=headers)
    assert response.status_code == 503
