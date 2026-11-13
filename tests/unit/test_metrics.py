"""Service metrics shared through Redis."""

from decimal import Decimal

from arq.connections import ArqRedis
from fakeredis import FakeAsyncRedis, FakeServer

from scambot.core.metrics import BUCKETS, HASH_KEY, JobMetrics, Metrics, render


def job() -> JobMetrics:
    jm = JobMetrics()
    jm.request = ("telegram", "text", "likely_scam")
    jm.checks = [("analyze_scam_patterns", "ok"), ("fact_check_claim", "error")]
    jm.cache_hits = ["checker", "verdict"]
    jm.cost_by_vendor = {
        "anthropic": Decimal("0.5"),
        "sightengine": Decimal("0.25"),
        "x": Decimal(0),
    }
    jm.stages = {"total": 0.3, "fetch": 100.0}
    return jm


async def test_all_five_families_are_rendered(redis: ArqRedis) -> None:
    await Metrics(redis).flush(job())

    lines = (await render(redis)).decode().splitlines()

    assert 'requests_total{channel="telegram",kind="text",label="likely_scam"} 1.0' in lines
    assert 'checker_runs_total{checker="fact_check_claim",status="error"} 1.0' in lines
    assert 'cache_hits_total{layer="verdict"} 1.0' in lines
    assert 'cache_hits_total{layer="checker"} 1.0' in lines
    assert 'api_cost_usd_total{vendor="anthropic"} 0.5' in lines
    assert 'api_cost_usd_total{vendor="sightengine"} 0.25' in lines
    assert not any('vendor="x"' in line for line in lines)  # free vendors add nothing


async def test_histogram_buckets_are_cumulative(redis: ArqRedis) -> None:
    metrics = Metrics(redis)
    await metrics.flush(job())
    await metrics.flush(job())

    lines = (await render(redis)).decode().splitlines()

    assert 'stage_seconds_bucket{le="0.1",stage="total"} 0.0' in lines
    assert 'stage_seconds_bucket{le="0.5",stage="total"} 2.0' in lines
    assert 'stage_seconds_bucket{le="90",stage="total"} 2.0' in lines
    assert 'stage_seconds_bucket{le="+Inf",stage="total"} 2.0' in lines
    assert 'stage_seconds_bucket{le="90",stage="fetch"} 0.0' in lines
    assert 'stage_seconds_bucket{le="+Inf",stage="fetch"} 2.0' in lines
    assert 'stage_seconds_count{stage="total"} 2.0' in lines
    assert 'stage_seconds_sum{stage="fetch"} 200.0' in lines
    assert len(BUCKETS) == 9


async def test_only_known_labels_are_rendered(redis: ArqRedis) -> None:
    await redis.hset(HASH_KEY, mapping={"c|requests_total|evil=1": "5", "junk": "x"})  # type: ignore[misc]

    text = (await render(redis)).decode()

    assert "evil" not in text
    assert "junk" not in text


async def test_two_processes_share_counts() -> None:
    server = FakeServer()
    api = ArqRedis(connection_pool=FakeAsyncRedis(server=server).connection_pool)
    worker = ArqRedis(connection_pool=FakeAsyncRedis(server=server).connection_pool)
    try:
        await Metrics(worker).flush(job())
        await Metrics(worker).flush(job())

        assert 'cache_hits_total{layer="verdict"} 2.0' in (await render(api)).decode()
    finally:
        await api.aclose(close_connection_pool=True)
        await worker.aclose(close_connection_pool=True)


async def test_a_broken_redis_never_raises() -> None:
    class Broken:
        def pipeline(self, **_kwargs: object) -> object:
            msg = "redis down"
            raise ConnectionError(msg)

    await Metrics(Broken()).flush(job())  # type: ignore[arg-type]


async def test_an_empty_job_writes_nothing(redis: ArqRedis) -> None:
    await Metrics(redis).flush(JobMetrics())

    assert await redis.exists(HASH_KEY) == 0


def test_timed_adds_up_the_time_of_a_stage() -> None:
    jm = JobMetrics()
    with jm.timed("fetch"):
        pass
    with jm.timed("fetch"):
        pass

    assert jm.stages["fetch"] >= 0
