"""A key-less stack for load runs, plus the request payload builders Locust uses.

Run it with ``uv run uvicorn eval.loadtest.fake_stack:app --port 8099``. It processes jobs with
the offline evaluation stack (fake vendors, real planner, verdict rules and reply composer) and
waits a configurable time per input type to stand in for vendor latency, so the numbers show how
the queueing and concurrency behave, not how fast the real vendors are.

This module must not import ``locust``; the tests use it directly.
"""

import asyncio
import base64
import io
import json
import os
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Final

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from eval.backends import Stack, build_offline_stack
from eval.replay import SEED_REPLAY, ReplayStore
from scambot.worker.pipeline import DEPS_KEY, process_job

# Milliseconds of simulated vendor time per input type (override with FAKE_LATENCY_MS as JSON).
DEFAULT_LATENCY_MS: Final = {
    "text": 300,
    "image": 1500,
    "audio": 3000,
    "voice": 3000,
    "video": 4000,
}
OGG_BYTES: Final = b"OggS" + b"\0" * 60
MP4_BYTES: Final = b"\0\0\0\x18ftypisom" + b"\0" * 60
WEBHOOK_SECRET_HEADER: Final = "X-Telegram-Bot-Api-Secret-Token"  # noqa: S105 - a header name
LOAD_TEXTS: Final = (
    "Your KYC expires today, update it at http://kyc-update.example.com/verify now",
    "Dear customer your electricity will be disconnected tonight, call +91 90000 00000",
    "Happy birthday Riya! See you at the party tomorrow evening.",
    "Forwarded: free tablets for every student this month, share with everyone",
)
MEDIA_KINDS: Final = {"image": "photo", "audio": "audio", "voice": "voice", "video": "video"}


def latency_from_env() -> dict[str, int]:
    raw = os.environ.get("FAKE_LATENCY_MS")
    if not raw:
        return dict(DEFAULT_LATENCY_MS)
    return {**DEFAULT_LATENCY_MS, **{k: int(v) for k, v in json.loads(raw).items()}}


def jpeg_bytes(seed: int = 0) -> bytes:
    """A small picture; the seed changes the colour so each upload is a different file."""
    image = Image.new("RGB", (64, 48), ((seed * 37) % 256, (seed * 91) % 256, (seed * 53) % 256))
    for x in range(0, 64, 8):
        for y in range(48):
            image.putpixel((x, y), (255 - (seed % 200), 40, 40 + seed % 150))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=60)
    return buffer.getvalue()


def media_bytes(kind: str, seed: int) -> bytes:
    if kind == "image":
        return jpeg_bytes(seed)
    return MP4_BYTES if kind == "video" else OGG_BYTES + seed.to_bytes(4, "big")


def make_payload(kind: str, number: int, *, user_id: int | None = None) -> dict[str, Any]:
    """An update in the format the fake stack (and the test fake channel) reads."""
    user = user_id if user_id is not None else 1_000_000 + number
    if kind == "text":
        text = LOAD_TEXTS[number % len(LOAD_TEXTS)] + f" #{number}"
        return {"update_id": number, "user_id": user, "text": text, "lang": "en"}
    ref = f"load-{kind}-{number}"
    return {
        "update_id": number,
        "user_id": user,
        "text": f"caption {number}",
        "lang": "en",
        "media": {
            "kind": kind,
            "ref": ref,
            "size": None,
            "mime": None,
            "source": MEDIA_KINDS[kind],
        },
    }


def make_telegram_update(kind: str, number: int) -> dict[str, Any]:
    """A Telegram update for a real instance (media needs a file the bot can fetch)."""
    user = 1_000_000 + number
    message: dict[str, Any] = {
        "message_id": number,
        "date": 1_700_000_000 + number,
        "chat": {"id": user, "type": "private"},
        "from": {"id": user, "is_bot": False, "first_name": "load", "language_code": "en"},
    }
    if kind == "text":
        message["text"] = LOAD_TEXTS[number % len(LOAD_TEXTS)] + f" #{number}"
    elif kind == "image":
        message["photo"] = [
            {"file_id": f"load-{number}", "file_unique_id": f"u{number}", "width": 64, "height": 48}
        ]
    elif kind in {"audio", "voice"}:
        message["voice"] = {
            "file_id": f"load-{number}",
            "file_unique_id": f"u{number}",
            "duration": 5,
        }
    else:
        message["video"] = {
            "file_id": f"load-{number}", "file_unique_id": f"u{number}",
            "width": 64, "height": 48, "duration": 5,
        }  # fmt: skip
    return {"update_id": number, "message": message}


class Submission(BaseModel):
    update: dict[str, Any]
    data_b64: str | None = None


class FakeLoadState:
    def __init__(self, stack: Stack, latency_ms: dict[str, int], max_jobs: int) -> None:
        self.stack = stack
        self.latency_ms = latency_ms
        self.gate = asyncio.Semaphore(max_jobs)
        self.done: dict[str, str] = {}
        self.failed: set[str] = set()
        self.tasks: set[asyncio.Task[None]] = set()
        self.rng = random.Random(7)  # noqa: S311 - jitter, not security

    async def run(self, job_id: str, job: Any) -> None:
        async with self.gate:
            wait = self.latency_ms.get(job.kind, 0) / 1000 * self.rng.uniform(0.8, 1.2)
            await asyncio.sleep(wait)
            try:
                await process_job({DEPS_KEY: self.stack.deps, "job_try": 3}, job)
            except Exception:
                self.failed.add(job_id)
                self.done[job_id] = ""
                return
        replies = [t for _, chat, t in self.stack.adapter.log if chat == job.chat_id]
        self.done[job_id] = replies[-1] if replies else ""


def create_fake_app(
    latency_ms: dict[str, int] | None = None, *, max_jobs: int = 10, stack: Stack | None = None
) -> FastAPI:
    """The load-test app; ``stack`` lets a test supply its own, else one is built at startup."""
    holder: dict[str, FakeLoadState] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        built = stack or await build_offline_stack(replay=ReplayStore(SEED_REPLAY, autosave=False))
        holder["state"] = FakeLoadState(built, latency_ms or latency_from_env(), max_jobs)
        try:
            yield
        finally:
            await built.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.post("/loadtest/submit")
    async def submit(body: Submission) -> dict[str, str]:
        from scambot.core.contracts import Job  # noqa: PLC0415

        state = holder["state"]
        job = state.stack.adapter.parse_update(
            body.update, trace_id=f"load-{body.update['update_id']}"
        )
        if not isinstance(job, Job):
            raise HTTPException(status_code=400, detail="not a job")
        if job.media_ref:
            data = base64.b64decode(body.data_b64 or "") or media_bytes(
                job.kind, len(job.media_ref)
            )
            state.stack.adapter.files[job.media_ref] = data
        task = asyncio.create_task(state.run(job.job_id, job))
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)
        return {"id": job.job_id}

    @app.get("/loadtest/status/{job_id}")
    async def status(job_id: str) -> dict[str, Any]:
        state = holder["state"]
        done = job_id in state.done
        body: dict[str, Any] = {"done": done, "reply": state.done.get(job_id)}
        if job_id in state.failed:
            body["failed"] = True
        return body

    return app


app = create_fake_app()
