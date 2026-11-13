"""Every command, notice and button in English, Hindi and Hinglish."""

import re

import pytest
from arq.connections import ArqRedis
from pydantic import SecretStr

from scambot.api.gate import Gate
from scambot.api.intake import Intake
from scambot.core.locales import Locales
from scambot.core.runtime import RuntimeConfig, RuntimeDefaults
from scambot.core.security import user_hash
from scambot.core.settings import DEV_USER_ID_PEPPER
from tests.fakes import (
    FakeAdapter,
    InMemorySettingsStore,
    InMemoryStore,
    fake_callback,
    fake_update,
)
from tests.unit.test_intake_buttons import verdict_for

DEVANAGARI = re.compile("[ऀ-ॿ]")
ME = user_hash(7, SecretStr(DEV_USER_ID_PEPPER))
SCENARIOS = [
    "start",
    "help",
    "privacy",
    "language_prompt",
    "language_en",
    "language_hi",
    "language_hinglish",
    "history_empty",
    "history_rows",
    "delete_ok",
    "delete_failed",
    "unknown",
    "rate_limited",
    "rejection",
    "button_details",
    "button_helpful",
    "button_wrong",
]


async def run(
    scenario: str, lang: str, adapter: FakeAdapter, redis: ArqRedis, locales: Locales
) -> str:
    await redis.flushall()
    store = InMemoryStore()
    # The language under test is the stored choice, so Hinglish is covered like the others.
    store.languages[ME] = lang
    gate = None
    if scenario == "rate_limited":
        runtime = RuntimeConfig(InMemorySettingsStore(), redis, RuntimeDefaults(1, 7), 60)
        gate = Gate(runtime, redis, 3600)
    intake = Intake(adapter, redis, locales, gate=gate, users=store)
    kind, _, detail = scenario.partition("_")
    if scenario == "rate_limited":
        await intake.handle(fake_update(1, "first message"))
        adapter.log.clear()
        await intake.handle(fake_update(2, "second message"))
    elif scenario == "rejection":
        await intake.handle({**fake_update(1, "x"), "reject": "too_large"})
    elif kind == "button":
        request_id = await verdict_for(store, "t:1", ME)
        prefix = {"details": "d", "helpful": "h", "wrong": "w"}[detail]
        await intake.handle(fake_callback(1, f"{prefix}:{request_id}", lang=None))
        text = adapter.log[-1][2] if adapter.log else ""
        return text or str(adapter.answers[-1])
    elif scenario == "history_rows":
        await verdict_for(store, "t:1", ME)
        await verdict_for(store, "t:2", ME)
        await intake.handle(fake_update(1, "/history"))
    elif scenario == "delete_failed":

        async def refuse(_user: str) -> int:
            msg = "database unavailable"
            raise ConnectionError(msg)

        store.delete_user = refuse  # type: ignore[assignment]
        await intake.handle(fake_update(1, "/delete"))
    elif kind == "language" and detail != "prompt":
        arg = {"en": "en", "hi": "hi", "hinglish": "hinglish"}[detail]
        await intake.handle(fake_update(1, f"/language {arg}"))
    elif scenario == "language_prompt":
        await intake.handle(fake_update(1, "/language"))
    else:
        name = {"history_empty": "history", "delete_ok": "delete", "unknown": "frobnicate"}.get(
            scenario, scenario
        )
        await intake.handle(fake_update(1, f"/{name}"))
    return adapter.log[-1][2]


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("lang", ["en", "hi", "hi-Latn"])
async def test_reply_is_filled_in_and_in_the_right_script(
    scenario: str, lang: str, adapter: FakeAdapter, redis: ArqRedis, locales: Locales
) -> None:
    text = await run(scenario, lang, adapter, redis, locales)

    assert text
    assert "{" not in text
    assert "}" not in text
    if scenario.startswith("language_") and scenario != "language_prompt":
        return  # the confirmation is in the language just chosen, checked separately
    assert bool(DEVANAGARI.search(text)) is (lang == "hi")


@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_english_and_hindi_replies_differ(
    scenario: str, redis: ArqRedis, locales: Locales
) -> None:
    english = await run(scenario, "en", FakeAdapter(), redis, locales)
    hindi = await run(scenario, "hi", FakeAdapter(), redis, locales)

    if scenario not in {"language_en", "language_hi", "language_hinglish"}:
        assert english != hindi


@pytest.mark.parametrize(
    ("choice", "language", "devanagari"),
    [("en", "en", False), ("hi", "hi", True), ("hinglish", "hi-Latn", False)],
)
async def test_the_confirmation_is_in_the_new_language(
    choice: str,
    language: str,
    devanagari: bool,
    adapter: FakeAdapter,
    redis: ArqRedis,
    locales: Locales,
) -> None:
    text = await run(
        f"language_{choice}", "en" if choice != "en" else "hi", adapter, redis, locales
    )

    assert bool(DEVANAGARI.search(text)) is devanagari
    assert locales.get(language).language.saved.split("{")[0] in text
