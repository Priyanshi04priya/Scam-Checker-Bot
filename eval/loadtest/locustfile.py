"""Locust users for the load test (run with the ``loadtest`` dependency group).

Fake mode (``LOADTEST_FAKE_VENDORS=1``): posts to ``eval.loadtest.fake_stack`` and measures the
whole time until the verdict is ready. Real mode: posts Telegram updates to ``/webhooks/telegram``
and records only the acknowledgement time; read the verdict latencies from the database with
``python -m eval.loadtest.summarise --database-url ... --since ...``.

Environment: ``LOADTEST_WEBHOOK_SECRET``, ``LOADTEST_TOTAL`` (stop after this many requests,
default 100), ``LOADTEST_POLL_S`` (status poll interval, default 0.25).
"""

import base64
import itertools
import os
import time

from locust import HttpUser, between, events, task
from locust.env import Environment

from eval.loadtest.fake_stack import (
    WEBHOOK_SECRET_HEADER,
    make_payload,
    make_telegram_update,
    media_bytes,
)

FAKE = os.environ.get("LOADTEST_FAKE_VENDORS") == "1"
TOTAL = int(os.environ.get("LOADTEST_TOTAL", "100"))
POLL_S = float(os.environ.get("LOADTEST_POLL_S", "0.25"))
SECRET = os.environ.get("LOADTEST_WEBHOOK_SECRET", "")
_counter = itertools.count(1)
_done = itertools.count(1)


_state: dict[str, object] = {}


@events.init.add_listener
def _remember_environment(environment: Environment, **_kw: object) -> None:
    _state["environment"] = environment


@events.request.add_listener
def _stop_after_total(name: str = "", **_kw: object) -> None:
    if not name.startswith(("e2e_", "webhook_")):  # not the submit and status-poll calls
        return
    environment = _state.get("environment")
    if next(_done) >= TOTAL and isinstance(environment, Environment) and environment.runner:
        environment.runner.quit()


class Checker(HttpUser):
    wait_time = between(0.5, 2.0)

    def _send(self, kind: str) -> None:
        number = next(_counter)
        if FAKE:
            self._fake(kind, number)
        else:
            self._real(kind, number)

    def _fake(self, kind: str, number: int) -> None:
        payload = make_payload(kind, number)
        body: dict[str, object] = {"update": payload}
        if kind != "text":
            body["data_b64"] = base64.b64encode(media_bytes(kind, number)).decode()
        started = time.perf_counter()
        response = self.client.post("/loadtest/submit", json=body, name=f"submit_{kind}")
        job_id = response.json().get("id") if response.ok else None
        error: Exception | None = None if job_id else RuntimeError("submit failed")
        while job_id and time.perf_counter() - started < 120:
            time.sleep(POLL_S)
            reply = self.client.get(f"/loadtest/status/{job_id}", name="status")
            if reply.ok and reply.json().get("done"):
                if reply.json().get("failed"):
                    error = RuntimeError("job failed")
                break
        else:
            error = error or RuntimeError("timeout")
        events.request.fire(
            request_type="E2E",
            name=f"e2e_{kind}",
            response_time=(time.perf_counter() - started) * 1000,
            response_length=0,
            exception=error,
            context={},
        )

    def _real(self, kind: str, number: int) -> None:
        self.client.post(
            "/webhooks/telegram",
            json=make_telegram_update(kind, number),
            headers={WEBHOOK_SECRET_HEADER: SECRET},
            name=f"webhook_{kind}",
        )

    @task(5)
    def text(self) -> None:
        self._send("text")

    @task(2)
    def image(self) -> None:
        self._send("image")

    @task(1)
    def audio(self) -> None:
        self._send("audio")

    @task(1)
    def video(self) -> None:
        self._send("video")
