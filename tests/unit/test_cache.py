"""The per-checker result cache."""

import datetime as dt
from typing import Any

import pytest
from arq.connections import ArqRedis

from scambot.agent.analyze import Analysis
from scambot.agent.executor import CheckInputs, cache_key, input_hash
from scambot.core.cache import CheckerCache
from scambot.core.contracts import CheckerName, CheckerResult, Preprocessed
from scambot.core.redis import checker_key

TTLS: dict[CheckerName, int] = {
    "analyze_scam_patterns": 100,
    "fact_check_claim": 200,
    "check_provenance": 300,
    "check_caption_context": 400,
    "detect_deepfake": 500,
    "detect_voice_clone": 600,
}


def result(status: str = "ok", name: str = "check_provenance") -> CheckerResult:
    return CheckerResult(
        checker=name,
        status=status,  # type: ignore[arg-type]
        signal="none",
        confidence=0.5,
        duration_ms=3,
        details={"llm_usage": []},
    )


@pytest.fixture
def cache(redis: ArqRedis) -> CheckerCache:
    return CheckerCache(redis, TTLS, 60)


async def test_an_ok_result_round_trips_with_its_ttl(cache: CheckerCache, redis: ArqRedis) -> None:
    await cache.put("check_provenance", "k", result())

    assert await cache.get("check_provenance", "k") == result()
    assert await redis.ttl(checker_key("check_provenance", "k")) == 300


async def test_an_inconclusive_result_is_kept_for_the_short_time(
    cache: CheckerCache, redis: ArqRedis
) -> None:
    await cache.put("check_provenance", "k", result("inconclusive"))
    await cache.put("analyze_scam_patterns", "k", result("inconclusive", "analyze_scam_patterns"))

    assert await redis.ttl(checker_key("check_provenance", "k")) == 60
    assert await redis.ttl(checker_key("analyze_scam_patterns", "k")) == 60


async def test_an_error_is_never_stored(cache: CheckerCache, redis: ArqRedis) -> None:
    await cache.put("check_provenance", "k", result("error"))

    assert await redis.exists(checker_key("check_provenance", "k")) == 0


async def test_a_stored_error_is_not_served(cache: CheckerCache, redis: ArqRedis) -> None:
    await redis.set(checker_key("check_provenance", "k"), result("error").model_dump_json())

    assert await cache.get("check_provenance", "k") is None


async def test_corrupt_json_is_a_miss(cache: CheckerCache, redis: ArqRedis) -> None:
    await redis.set(checker_key("check_provenance", "k"), b"{not json")

    assert await cache.get("check_provenance", "k") is None


async def test_a_result_for_another_checker_is_a_miss(cache: CheckerCache, redis: ArqRedis) -> None:
    await redis.set(
        checker_key("check_provenance", "k"), result(name="detect_deepfake").model_dump_json()
    )

    assert await cache.get("check_provenance", "k") is None


async def test_redis_being_down_is_a_miss_and_a_silent_skip() -> None:
    class Broken:
        async def get(self, *_args: Any) -> None:
            raise ConnectionError

        async def set(self, *_args: Any, **_kwargs: Any) -> None:
            raise ConnectionError

    cache = CheckerCache(Broken(), TTLS, 60)  # type: ignore[arg-type]

    assert await cache.get("check_provenance", "k") is None
    await cache.put("check_provenance", "k", result())


def provenance_args(reference: str) -> dict[str, Any]:
    return {"media_key": "m", "reference_date": reference, "sha256": "s", "frames": ["original"]}


def test_reference_date_is_dropped_without_a_claimed_date() -> None:
    pre = Preprocessed(kind="image")
    inputs = CheckInputs(pre=pre, analysis=None, trace_id="t")

    first = cache_key("check_provenance", provenance_args("2026-11-13"), inputs)
    second = cache_key("check_provenance", provenance_args("2026-11-14"), inputs)

    assert first == second
    # input_hash itself, which is stored with each run, still includes it.
    assert input_hash(provenance_args("2026-11-13")) != input_hash(provenance_args("2026-11-14"))


def test_reference_date_is_kept_with_a_claimed_date() -> None:
    analysis = Analysis(
        claims=[],
        claimed_date=dt.date(2020, 1, 1),
        reply_language="en",
        claimed_place=None,
        extra_checkers=[],
        reason="",
    )
    inputs = CheckInputs(pre=Preprocessed(kind="image"), analysis=analysis, trace_id="t")

    assert cache_key("check_provenance", provenance_args("2020-01-01"), inputs) != cache_key(
        "check_provenance", provenance_args("2020-01-02"), inputs
    )


def test_other_checkers_use_the_plain_input_hash() -> None:
    inputs = CheckInputs(pre=Preprocessed(kind="image"), analysis=None, trace_id="t")
    args = {"frames": ["original"], "sha256": "s", "reference_date": "x"}

    assert cache_key("detect_deepfake", args, inputs) == input_hash(args)
