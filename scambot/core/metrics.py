"""Service metrics shared by separate processes through one Redis hash.

The API, the worker and the checker server run as different processes, so a per-process
Prometheus registry would show only a fraction of the traffic. Each worker adds its counts
to a Redis hash; the API reads the hash and renders it in the Prometheus text format.
"""

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final

from arq.connections import ArqRedis
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, HistogramMetricFamily, Metric

from scambot.core.logging import get_logger

BUCKETS: Final = (0.1, 0.5, 1, 2, 5, 10, 20, 40, 90)
HASH_KEY: Final = "metrics:v1"
CONTENT_TYPE: Final = "text/plain; version=0.0.4; charset=utf-8"

REQUESTS: Final = "requests_total"
STAGE_SECONDS: Final = "stage_seconds"
CHECKER_RUNS: Final = "checker_runs_total"
CACHE_HITS: Final = "cache_hits_total"
API_COST: Final = "api_cost_usd_total"

_COUNTERS: Final = {
    REQUESTS: ("Requests answered.", ("channel", "kind", "label")),
    CHECKER_RUNS: ("Checker runs by outcome.", ("checker", "status")),
    CACHE_HITS: ("Answers or checks served from a cache.", ("layer",)),
    API_COST: ("Estimated spend on paid services in US dollars.", ("vendor",)),
}
_STAGE_LABELS: Final = ("stage",)

_log = get_logger(component="metrics")


@dataclass
class JobMetrics:
    """What one job measured; added to the shared counters once, when the job ends."""

    stages: dict[str, float] = field(default_factory=dict)
    checks: list[tuple[str, str]] = field(default_factory=list)
    cache_hits: list[str] = field(default_factory=list)
    cost_by_vendor: dict[str, Decimal] = field(default_factory=dict)
    request: tuple[str, str, str] | None = None

    @contextmanager
    def timed(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[stage] = self.stages.get(stage, 0.0) + time.perf_counter() - start


def _labels(**labels: str) -> str:
    return ",".join(f"{key}={value}" for key, value in labels.items())


def _bucket(seconds: float) -> str:
    for edge in BUCKETS:
        if seconds <= edge:
            return f"{edge:g}"
    return "+Inf"


def _increments(jm: JobMetrics) -> dict[str, float]:
    """Hash field → amount to add."""
    adds: dict[str, float] = {}

    def add(field_name: str, amount: float = 1.0) -> None:
        adds[field_name] = adds.get(field_name, 0.0) + amount

    if jm.request is not None:
        channel, kind, label = jm.request
        add(f"c|{REQUESTS}|{_labels(channel=channel, kind=kind, label=label)}")
    for checker, status in jm.checks:
        add(f"c|{CHECKER_RUNS}|{_labels(checker=checker, status=status)}")
    for layer in jm.cache_hits:
        add(f"c|{CACHE_HITS}|{_labels(layer=layer)}")
    for vendor, cost in jm.cost_by_vendor.items():
        if cost > 0:
            add(f"c|{API_COST}|{_labels(vendor=vendor)}", float(cost))
    for stage, seconds in jm.stages.items():
        base = f"h|{STAGE_SECONDS}|{_labels(stage=stage)}"
        add(f"{base}|le={_bucket(seconds)}")
        add(f"{base}|sum", seconds)
        add(f"{base}|count")
    return adds


class Metrics:
    """Writes job metrics to the shared hash; ``None`` Redis problems never reach the caller."""

    def __init__(self, redis: ArqRedis) -> None:
        self._redis = redis

    async def flush(self, jm: JobMetrics) -> None:
        adds = _increments(jm)
        if not adds:
            return
        try:
            async with self._redis.pipeline(transaction=False) as pipe:
                for name, amount in adds.items():
                    pipe.hincrbyfloat(HASH_KEY, name, amount)
                await pipe.execute()
        except Exception as exc:  # metrics are best effort
            _log.warning("metrics_failed", error=type(exc).__name__)


def _parse_labels(text: str, names: tuple[str, ...]) -> tuple[str, ...] | None:
    pairs = dict(item.split("=", 1) for item in text.split(",") if "=" in item)
    if tuple(pairs) != names:
        return None
    return tuple(pairs.values())


class _Collector:
    def __init__(self, hash_values: dict[str, float]) -> None:
        self._values = hash_values

    def collect(self) -> Iterable[Metric]:
        yield from self._counters()
        yield from self._histograms()

    def _counters(self) -> Iterator[Metric]:
        for name, (doc, names) in _COUNTERS.items():
            family = CounterMetricFamily(name.removesuffix("_total"), doc, labels=list(names))
            rows: list[tuple[tuple[str, ...], float]] = []
            for key, value in self._values.items():
                kind, _, rest = key.partition("|")
                metric, _, label_text = rest.partition("|")
                if kind != "c" or metric != name:
                    continue
                if (labels := _parse_labels(label_text, names)) is not None:
                    rows.append((labels, value))
            for labels, value in sorted(rows):
                family.add_metric(list(labels), value)
            yield family

    def _histograms(self) -> Iterator[Metric]:
        family = HistogramMetricFamily(
            STAGE_SECONDS, "Time spent per processing stage.", labels=list(_STAGE_LABELS)
        )
        stages: dict[str, dict[str, float]] = {}
        for key, value in self._values.items():
            parts = key.split("|")
            if len(parts) != 4 or parts[0] != "h" or parts[1] != STAGE_SECONDS:
                continue
            labels = _parse_labels(parts[2], _STAGE_LABELS)
            if labels is not None:
                stages.setdefault(labels[0], {})[parts[3]] = value
        for stage, fields in sorted(stages.items()):
            running = 0.0
            buckets: list[tuple[str, float]] = []
            for edge in BUCKETS:
                running += fields.get(f"le={edge:g}", 0.0)
                buckets.append((f"{edge:g}", running))
            running += fields.get("le=+Inf", 0.0)
            buckets.append(("+Inf", running))
            family.add_metric([stage], buckets, fields.get("sum", 0.0))
        yield family


async def render(redis: ArqRedis) -> bytes:
    """The shared counters in the Prometheus text format; raises when Redis is unreachable."""
    raw = await redis.hgetall(HASH_KEY)  # type: ignore[misc]
    values: dict[str, float] = {}
    for key, value in raw.items():
        try:
            values[key.decode() if isinstance(key, bytes) else key] = float(value)
        except ValueError:
            continue
    registry = CollectorRegistry()
    registry.register(_Collector(values))
    return generate_latest(registry)
