"""The load-test helpers that need no Locust: the summary and the in-process fake stack."""

import asyncio
import base64
import datetime as dt
from pathlib import Path

import httpx
import pytest
from eval.backends import build_offline_stack
from eval.loadtest import fake_stack, summarise
from eval.replay import SEED_REPLAY, ReplayStore
from sqlalchemy.ext.asyncio import async_sessionmaker

from scambot.core.models import Request, User
from tests.fakes import FakeAdapter, fake_media_update, fake_update

CSV = Path(__file__).parents[1] / "fixtures" / "loadtest_stats.csv"


def test_summarise_parses_and_grades_the_canned_csv() -> None:
    lines = {ln.name: ln for ln in summarise.parse_stats(CSV)}
    assert set(lines) == {"text", "image", "audio", "video"}
    assert lines["text"].status == "PASS"
    assert lines["text"].p95_s == pytest.approx(0.48)
    assert lines["audio"].failures == 1
    assert lines["audio"].status == "FAIL"  # a failed request and a 40 s max
    text = summarise.render(list(lines.values()))
    assert "| text | 50 | 0 |" in text
    assert "FAIL" in text


def test_line_status_cases() -> None:
    ok = summarise.Line("x", 10, 0, 1, 2, 3, 15)
    assert ok.status == "PASS"
    assert summarise.Line("x", 10, 0, 1, 20, 30, 15).status == "FAIL"
    assert summarise.Line("x", 10, 0, 1, 2, 95, 15).status == "FAIL"
    assert summarise.Line("x", 0, 0, 0, 0, 0, 15).status == "INCOMPLETE"


def test_lines_from_database_durations() -> None:
    lines = summarise.lines_from_durations({"text": [1000, 2000, 3000], "image": [40000]})
    by_name = {ln.name: ln for ln in lines}
    assert by_name["verdict text"].status == "PASS"
    assert by_name["verdict image"].status == "FAIL"


def test_cli_writes_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "s.md"
    assert summarise.main(["--csv", str(CSV), "--out", str(out)]) == 0
    assert "Load test summary" in out.read_text("utf-8")
    assert "e2e" not in capsys.readouterr().out
    assert summarise.main(["--csv", str(tmp_path / "missing.csv")]) == 2
    assert summarise.main(["--csv", str(CSV), "--database-url", "sqlite://"]) == 2


async def test_read_durations_from_database(sessions: object, tmp_path: Path) -> None:
    assert isinstance(sessions, async_sessionmaker)
    async with sessions() as session:
        session.add(User(user_hash="u" * 8))
        session.add(Request(job_id="j1", user_hash="u" * 8, channel="telegram", kind="text",
                            status="replied", duration_ms=2500))  # fmt: skip
        await session.commit()
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    since = (dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).isoformat()
    assert await summarise.read_durations(url, since) == {"text": [2500]}


def test_payload_builders_make_valid_updates() -> None:
    adapter = FakeAdapter()
    for kind in ("text", "image", "audio", "voice", "video"):
        payload = fake_stack.make_payload(kind, 5)
        job = adapter.parse_update(payload, trace_id="t")
        assert job is not None
        assert getattr(job, "kind", None) == kind
    assert (
        fake_stack.make_payload("text", 1)["update_id"]
        != fake_stack.make_payload("text", 2)["update_id"]
    )
    assert fake_stack.make_payload("text", 1) == fake_update(
        1, fake_stack.make_payload("text", 1)["text"], user_id=1_000_001
    )
    media = fake_stack.make_payload("image", 3)
    assert media["media"]["kind"] == fake_media_update(3, "image", "x")["media"]["kind"]
    assert fake_stack.jpeg_bytes(1) != fake_stack.jpeg_bytes(2)
    tg = fake_stack.make_telegram_update("image", 4)
    assert tg["message"]["photo"][0]["file_id"] == "load-4"
    assert "text" in fake_stack.make_telegram_update("text", 4)["message"]
    assert "voice" in fake_stack.make_telegram_update("audio", 4)["message"]
    assert "video" in fake_stack.make_telegram_update("video", 4)["message"]


def test_latency_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert fake_stack.latency_from_env()["text"] == 300
    monkeypatch.setenv("FAKE_LATENCY_MS", '{"text": 5}')
    assert fake_stack.latency_from_env()["text"] == 5
    assert fake_stack.latency_from_env()["image"] == 1500


async def test_fake_stack_round_trip() -> None:
    stack = await build_offline_stack(replay=ReplayStore(SEED_REPLAY, autosave=False))
    app = fake_stack.create_fake_app({"text": 1, "image": 1}, stack=stack)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            text = await client.post(
                "/loadtest/submit", json={"update": fake_stack.make_payload("text", 1)}
            )
            image = await client.post(
                "/loadtest/submit",
                json={
                    "update": fake_stack.make_payload("image", 2),
                    "data_b64": base64.b64encode(fake_stack.jpeg_bytes(2)).decode(),
                },
            )
            bad = await client.post(
                "/loadtest/submit",
                json={"update": {"update_id": 9, "user_id": 1, "text": "/start"}},
            )
            assert bad.status_code == 400
            ids = [text.json()["id"], image.json()["id"]]
            for _ in range(200):
                states = [(await client.get(f"/loadtest/status/{i}")).json() for i in ids]
                if all(s["done"] for s in states):
                    break
                await asyncio.sleep(0.05)
            assert all(s["done"] and s["reply"] for s in states)
            assert (await client.get("/loadtest/status/unknown")).json() == {
                "done": False,
                "reply": None,
            }
