"""The admin pages: login, request-forgery protection, numbers, controls and rendering."""

import datetime as dt
import hashlib
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import httpx
import pytest
from argon2 import PasswordHasher
from arq.connections import ArqRedis
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.api.admin.auth import FAILURE_WINDOW_S, MAX_FAILURES, AdminAuth
from scambot.api.admin.deps import AdminDeps
from scambot.api.admin.queries import StatsStore, day_bucket
from scambot.api.admin.routes import STATIC_DIR
from scambot.api.gate import Admission, Gate
from scambot.api.main import create_app
from scambot.core.config import AppConfig
from scambot.core.metrics import JobMetrics, Metrics
from scambot.core.models import CheckerRun, Feedback, Request, User, VerdictRow
from scambot.core.runtime import (
    SOURCES_KEY,
    RuntimeConfig,
    RuntimeDefaults,
    SqlSettingsStore,
    effective_sources,
)
from scambot.core.settings import Settings
from scambot.factcheck_index.ingest import IngestDeps
from scambot.worker import cron

HTMX_SHA256 = "e209dda5c8235479f3166defc7750e1dbcd5a5c1808b7792fc2e6733768fb447"
NOW = dt.datetime(2026, 11, 13, 12, 0, tzinfo=dt.UTC)  # 17:30 in India
PASSWORD = "correct horse"
HASH = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash(PASSWORD)
GOOD = httpx.BasicAuth("admin", PASSWORD)
GET_ROUTES = ["/admin/", "/admin/fragments/stats", "/admin/controls", "/admin/sources", "/metrics"]
Sessions = async_sessionmaker[AsyncSession]
SECRET_TEXT = "SECRET-MESSAGE-TEXT"
USERS = [f"{i:x}" * 2 + "ab" * 30 for i in range(3)]  # 64 hex characters each


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> dt.datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def deps(sessions: Sessions, redis: ArqRedis, app_config: AppConfig, clock: Clock) -> AdminDeps:
    store = SqlSettingsStore(sessions)
    runtime = RuntimeConfig(store, redis, RuntimeDefaults(20, 7), 60)
    return AdminDeps(StatsStore(sessions), store, runtime, redis, app_config, now=clock)


def make_app(deps: AdminDeps, *, password_hash: str = HASH) -> Any:
    settings = Settings(_env_file=None, admin_password_hash=SecretStr(password_hash))
    return create_app(settings, telegram_intake=object(), admin=deps)  # type: ignore[arg-type]


@pytest.fixture
async def client(deps: AdminDeps) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=make_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


async def token(client: httpx.AsyncClient) -> str:
    page = await client.get("/admin/controls", auth=GOOD)
    marker = 'name="csrf_token" value="'
    start = page.text.index(marker) + len(marker)
    return page.text[start : page.text.index('"', start)]


# --- login ---------------------------------------------------------------------------------


@pytest.mark.parametrize("path", GET_ROUTES)
async def test_login_is_required(client: httpx.AsyncClient, path: str) -> None:
    missing = await client.get(path)
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"].startswith("Basic")
    assert (await client.get(path, auth=httpx.BasicAuth("admin", "nope"))).status_code == 401
    assert (await client.get(path, auth=httpx.BasicAuth("root", PASSWORD))).status_code == 401
    assert (await client.get(path, auth=GOOD)).status_code == 200


async def test_garbage_headers_are_refused(client: httpx.AsyncClient) -> None:
    for header in ("Basic !!!", "Bearer abc", "Basic "):
        response = await client.get("/admin/", headers={"Authorization": header})
        assert response.status_code == 401


async def test_a_broken_hash_in_settings_is_a_refusal_not_an_error(deps: AdminDeps) -> None:
    transport = httpx.ASGITransport(app=make_app(deps, password_hash="not-a-hash"))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
        assert (await http.get("/admin/", auth=GOOD)).status_code == 401


async def test_without_a_hash_nothing_is_mounted(deps: AdminDeps) -> None:
    transport = httpx.ASGITransport(app=make_app(deps, password_hash=""))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
        assert (await http.get("/admin/", auth=GOOD)).status_code == 404
        assert (await http.get("/metrics", auth=GOOD)).status_code == 404
        assert (await http.get("/health")).status_code == 200


async def test_successful_logins_are_remembered_failures_are_not() -> None:
    now = [0.0]
    auth = AdminAuth("admin", HASH, "p", cache_s=300, csrf_ttl_s=60, clock=lambda: now[0])
    header = httpx.BasicAuth("admin", PASSWORD)._auth_header
    bad = httpx.BasicAuth("admin", "x")._auth_header

    assert await auth.check(header)
    auth._hash = "broken"  # a remembered header no longer needs the hash
    assert await auth.check(header)
    now[0] = 301
    assert not await auth.check(header)
    assert not await auth.check(bad)
    assert not await auth.check(bad)


async def test_the_memory_of_logins_is_bounded() -> None:
    auth = AdminAuth("admin", HASH, "p", cache_s=300, csrf_ttl_s=60)
    for i in range(20):
        auth._ok[str(i)] = 10**9
    assert await auth.check(httpx.BasicAuth("admin", PASSWORD)._auth_header)
    assert len(auth._ok) <= 16


async def test_security_headers(client: httpx.AsyncClient) -> None:
    for response in (await client.get("/admin/"), await client.get("/admin/", auth=GOOD)):
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "default-src 'self'" in response.headers["content-security-policy"]


# --- request forgery -----------------------------------------------------------------------


async def post(
    client: httpx.AsyncClient, form: dict[str, str], headers: dict[str, str] | None = None
) -> httpx.Response:
    return await client.post("/admin/settings", data=form, headers=headers, auth=GOOD)


async def test_posts_need_a_valid_token(
    client: httpx.AsyncClient, clock: Clock, deps: AdminDeps
) -> None:
    good = await token(client)
    form = {"rate_limit_per_hour": "5", "cache_window_days": ""}

    assert (await post(client, form)).status_code == 403
    assert (await post(client, form, {"X-CSRF-Token": good + "0"})).status_code == 403
    assert (await post(client, form, {"X-CSRF-Token": "x.y"})).status_code == 403
    assert (await post(client, {**form, "csrf_token": "\u00b2.\u00e9"})).status_code == 403
    forged = {"X-CSRF-Token": good, "Origin": "http://evil.example"}
    assert (await post(client, form, forged)).status_code == 403
    assert await deps.settings_store.get("rate_limit_per_hour") is None

    assert (await post(client, form, {"X-CSRF-Token": good})).status_code == 303
    assert await deps.settings_store.get("rate_limit_per_hour") == 5
    same_site = {"X-CSRF-Token": good, "Origin": "http://testserver"}
    assert (await post(client, form, same_site)).status_code == 303
    assert (await post(client, {**form, "csrf_token": good})).status_code == 303

    clock.now = NOW + dt.timedelta(hours=3)
    assert (await post(client, form, {"X-CSRF-Token": good})).status_code == 403
    fresh = await token(client)  # a swapped-in form carries a fresh token beside the stale header
    assert (
        await post(client, {**form, "csrf_token": fresh}, {"X-CSRF-Token": good})
    ).status_code == 303


async def test_a_token_belongs_to_the_password() -> None:
    one = AdminAuth("admin", HASH, "p", cache_s=1, csrf_ttl_s=60)
    other = AdminAuth("admin", HASH, "q", cache_s=1, csrf_ttl_s=60)

    assert one.token_valid(one.make_token(100), 100)
    assert not other.token_valid(one.make_token(100), 100)
    assert not one.token_valid(one.make_token(100), 99)  # from the future


async def test_reading_pages_changes_nothing(client: httpx.AsyncClient, deps: AdminDeps) -> None:
    for path in GET_ROUTES:
        await client.get(path + "?rate_limit_per_hour=1&blocked=true", auth=GOOD)

    assert await deps.settings_store.get("rate_limit_per_hour") is None
    assert await deps.stats.blocked_users() == []


# --- numbers -------------------------------------------------------------------------------


async def seed(sessions: Sessions) -> None:
    """Three days: 2 requests two days ago, 1 yesterday, 3 today (India time)."""
    spec = [
        # days_ago, hour (UTC), user, kind, label, duration, cost, cached, checker status
        (2, 10, 0, "text", "likely_scam", 1000, "0.04", False, "ok"),
        (2, 11, 0, "image", "no_issues", 3000, "0.06", False, "error"),
        (1, 9, 1, "text", "likely_scam", 2000, "0.02", False, "ok"),
        (0, 1, 1, "text", "likely_scam", 100, "0", True, "ok"),
        (0, 2, 2, "text", "false_misleading", 4000, "0.08", False, "error"),
        (0, 3, 2, "text", "no_issues", 1000, "0.00", False, "ok"),
    ]
    async with sessions() as session, session.begin():
        for hash_ in USERS:
            session.add(User(user_hash=hash_))
        await session.flush()
        previous: Request | None = None
        for i, (ago, hour, user, kind, label, ms, cost, cached, status) in enumerate(spec):
            when = (NOW - dt.timedelta(days=ago)).replace(hour=hour)
            row = Request(
                id=uuid.uuid4(),
                job_id=f"job-{i}",
                user_hash=USERS[user],
                channel="telegram",
                kind=kind,
                claims=[SECRET_TEXT],
                status="replied",
                created_at=when,
                replied_at=when,
                duration_ms=ms,
                cost_usd=Decimal(cost),
                cache_of=previous.id if cached and previous else None,
            )
            session.add(row)
            await session.flush()
            session.add(
                VerdictRow(
                    request_id=row.id,
                    label=label,
                    band="high",
                    score=90,
                    reply_text=SECRET_TEXT,
                    reply_language="en",
                )
            )
            session.add(
                CheckerRun(
                    request_id=row.id,
                    checker="detect_deepfake" if kind == "image" else "analyze_scam_patterns",
                    input_hash="h",
                    status=status,
                    signal="none",
                    confidence=0.0,
                    evidence=[],
                    details={"secret": SECRET_TEXT},
                    duration_ms=1,
                    error="timeout" if status == "error" else None,
                )
            )
            previous = row
            if i in (0, 2, 4):
                session.add(Feedback(request_id=row.id, user_hash=USERS[user], helpful=i == 0))


async def test_overview_numbers(deps: AdminDeps, sessions: Sessions) -> None:
    await seed(sessions)

    o = await deps.stats.overview(3, NOW)

    assert o.per_day == [("2026-11-11", 2), ("2026-11-12", 1), ("2026-11-13", 3)]
    assert o.total == 6
    assert [(v.label, v.count) for v in o.verdicts] == [
        ("likely_scam", 3),
        ("no_issues", 2),
        ("false_misleading", 1),
    ]
    assert o.verdicts[0].share == 0.5
    assert o.latency.overall_ms == pytest.approx(11100 / 6)
    assert o.latency.cache_ms == 100
    assert o.latency.fresh_ms == pytest.approx(11000 / 5)
    assert o.latency.by_kind == {"image": 3000, "text": pytest.approx(8100 / 5)}
    rates = {c.checker: (c.total, c.errors) for c in o.checkers}
    assert rates == {"analyze_scam_patterns": (5, 1), "detect_deepfake": (1, 1)}
    assert o.checkers[1].top_errors == ["timeout"]
    assert o.feedback.helpful == 1
    assert o.feedback.wrong == 2
    assert o.feedback.wrong_by_label["false_misleading"] == (1, 1)
    assert float(o.average_cost) == pytest.approx(0.2 / 6)
    assert o.blocked_users == 0
    assert (await deps.stats.overview(1, NOW)).per_day == [("2026-11-13", 3)]


def test_day_buckets_for_both_databases() -> None:
    def sql(dialect: str) -> str:
        expression = day_bucket(dialect, Request.created_at)
        return str(expression.compile(compile_kwargs={"literal_binds": True}))

    assert "'Asia/Kolkata'" in sql("postgresql")
    assert "'+5 hours', '+30 minutes'" in sql("sqlite")


async def test_empty_database_renders_zeros(client: httpx.AsyncClient) -> None:
    page = await client.get("/admin/fragments/stats", auth=GOOD)

    assert page.status_code == 200
    assert "No checks yet" in page.text


async def test_days_are_clamped(client: httpx.AsyncClient) -> None:
    for days, shown in (("0", "1"), ("9999", "90"), ("7", "7")):
        page = await client.get(f"/admin/fragments/stats?days={days}", auth=GOOD)
        assert page.text.count('<text x="0"') == int(shown)


async def test_fragment_shows_the_numbers_and_flags_cost_over_budget(
    client: httpx.AsyncClient, sessions: Sessions
) -> None:
    await seed(sessions)

    page = await client.get("/admin/fragments/stats?days=3", auth=GOOD)

    assert "$0.0333" in page.text
    assert "likely_scam" in page.text
    assert 'class="tile"' in page.text
    assert "tile over" not in page.text


# --- rendering safety ----------------------------------------------------------------------


async def test_no_user_content_appears_in_any_page(
    client: httpx.AsyncClient, sessions: Sessions
) -> None:
    await seed(sessions)
    for path in ("/admin/", "/admin/fragments/stats?days=3", "/admin/controls", "/admin/sources"):
        text = (await client.get(path, auth=GOOD)).text
        assert SECRET_TEXT not in text
        assert all(full not in text for full in USERS)
    controls = (await client.get("/admin/controls", auth=GOOD)).text
    assert USERS[2][:12] in controls


async def test_pages_use_no_third_party_hosts(client: httpx.AsyncClient) -> None:
    for path in ("/admin/", "/admin/controls", "/admin/sources"):
        text = (await client.get(path, auth=GOOD)).text
        assert "http://" not in text
        assert "https://" not in text.replace('placeholder="https', "")


def test_the_bundled_script_is_the_pinned_one() -> None:
    digest = hashlib.sha256((STATIC_DIR / "htmx.min.js").read_bytes()).hexdigest()
    assert digest == HTMX_SHA256


async def test_static_files_are_served(client: httpx.AsyncClient) -> None:
    assert (await client.get("/admin/static/htmx.min.js")).status_code == 200
    assert (await client.get("/admin/static/admin.css")).status_code == 200


# --- limits --------------------------------------------------------------------------------


async def test_a_new_limit_applies_at_once(
    client: httpx.AsyncClient, deps: AdminDeps, redis: ArqRedis
) -> None:
    gate = Gate(deps.runtime, redis, 3600, clock=deps.now)
    assert (await gate.admit("u")).admission is Admission.ALLOWED
    assert (await deps.runtime.rate_limit()) == 20  # now cached
    await redis.delete("rl:" + "u" + ":" + NOW.strftime("%Y%m%d%H"))
    good = await token(client)

    response = await post(client, {"rate_limit_per_hour": "2"}, {"X-CSRF-Token": good})
    assert response.status_code == 303
    admissions = [(await gate.admit("v")).admission for _ in range(3)]

    assert admissions == [Admission.ALLOWED, Admission.ALLOWED, Admission.LIMITED]
    await post(client, {"rate_limit_per_hour": ""}, {"X-CSRF-Token": good})
    assert await deps.runtime.rate_limit() == 20
    assert await deps.settings_store.get("rate_limit_per_hour") is None


@pytest.mark.parametrize(
    "form",
    [
        {"rate_limit_per_hour": "0"},
        {"rate_limit_per_hour": "5000"},
        {"rate_limit_per_hour": "ten"},
        {"rate_limit_per_hour": "2.5"},
        {"rate_limit_per_hour": "nan"},
        {"cache_window_days": "0"},
        {"cache_window_days": "400"},
        {"rate_limit_per_hour": "3", "cache_window_days": "x"},
    ],
)
async def test_invalid_limits_are_refused_without_a_write(
    client: httpx.AsyncClient, deps: AdminDeps, form: dict[str, str]
) -> None:
    response = await post(client, form, {"X-CSRF-Token": await token(client)})

    assert response.status_code == 422
    assert await deps.settings_store.get("rate_limit_per_hour") is None
    assert await deps.settings_store.get("cache_window_days") is None


async def test_the_controls_page_names_where_each_value_comes_from(
    client: httpx.AsyncClient,
) -> None:
    await post(client, {"cache_window_days": "0.5"}, {"X-CSRF-Token": await token(client)})

    page = (await client.get("/admin/controls", auth=GOOD)).text

    assert "from table" in page
    assert "from file" in page


async def test_htmx_gets_the_refreshed_fragment(client: httpx.AsyncClient) -> None:
    response = await post(
        client,
        {"rate_limit_per_hour": "7"},
        {"X-CSRF-Token": await token(client), "HX-Request": "true"},
    )

    assert response.status_code == 200
    assert "<html" not in response.text
    assert 'value="7"' in response.text


# --- blocking ------------------------------------------------------------------------------


async def block(client: httpx.AsyncClient, prefix: str, blocked: bool = True) -> httpx.Response:
    return await client.post(
        "/admin/users/block",
        data={"prefix": prefix, "blocked": str(blocked).lower()},
        headers={"X-CSRF-Token": await token(client)},
        auth=GOOD,
    )


async def test_block_and_unblock_apply_at_once(
    client: httpx.AsyncClient, deps: AdminDeps, sessions: Sessions, redis: ArqRedis
) -> None:
    await seed(sessions)
    gate = Gate(deps.runtime, redis, 3600, clock=deps.now)
    assert (await gate.admit(USERS[0])).admission is Admission.ALLOWED  # warms the cache

    assert (await block(client, USERS[0][:12])).status_code == 303
    assert (await gate.admit(USERS[0])).admission is Admission.BLOCKED
    assert await deps.stats.blocked_users() == [USERS[0]]
    assert "blocked" in (await client.get("/admin/controls", auth=GOOD)).text

    assert (await block(client, USERS[0][:12], False)).status_code == 303
    assert (await gate.admit(USERS[0])).admission is Admission.ALLOWED


async def test_bad_prefixes_are_refused(client: httpx.AsyncClient, sessions: Sessions) -> None:
    await seed(sessions)
    async with sessions() as session, session.begin():
        session.add(User(user_hash=USERS[0][:20] + "0" * 44))

    assert (await block(client, "abc")).status_code == 422
    assert (await block(client, "zzzzzzzzzz")).status_code == 422
    assert (await block(client, "deadbeef")).status_code == 404
    assert (await block(client, USERS[0][:20])).status_code == 409


# --- sources -------------------------------------------------------------------------------


async def source(client: httpx.AsyncClient, **fields: str) -> httpx.Response:
    form = {"source_id": "newsite", "name": "News Site", "site": "newsite.org", "language": "en"}
    return await client.post(
        "/admin/sources/add",
        data={**form, **fields},
        headers={"X-CSRF-Token": await token(client)},
        auth=GOOD,
    )


async def test_sources_can_be_added_removed_and_reset(
    client: httpx.AsyncClient, deps: AdminDeps
) -> None:
    baseline = deps.config.sources
    assert await effective_sources(deps.settings_store, baseline) is baseline

    added = await source(client, feeds="https://newsite.org/feed\n", api="on")
    assert added.status_code == 303
    now = await effective_sources(deps.settings_store, baseline)
    assert [s.id for s in now.sources] == [*(s.id for s in baseline.sources), "newsite"]
    assert now.sources[-1].feeds == ["https://newsite.org/feed"]
    assert now.ratings == baseline.ratings
    assert "Using the list saved here" in (await client.get("/admin/sources", auth=GOOD)).text

    removed = await client.post(
        "/admin/sources/remove",
        data={"source_id": "newsite"},
        headers={"X-CSRF-Token": await token(client)},
        auth=GOOD,
    )
    assert removed.status_code == 303
    assert [s.id for s in (await effective_sources(deps.settings_store, baseline)).sources] == [
        s.id for s in baseline.sources
    ]

    await source(client)
    reset = await client.post(
        "/admin/sources/reset", headers={"X-CSRF-Token": await token(client)}, auth=GOOD
    )
    assert reset.status_code == 303
    assert await effective_sources(deps.settings_store, baseline) is baseline


@pytest.mark.parametrize(
    "fields",
    [
        {"site": "not a domain"},
        {"site": "127.0.0.1", "feeds": "https://127.0.0.1/feed"},
        {"site": "intranet.localhost"},
        {"site": "metadata.internal", "feeds": "https://metadata.internal/feed"},
        {"feeds": "http://newsite.org/feed"},
        {"feeds": "https://other.example/feed"},
        {"source_id": "BAD ID"},
        {"language": "english"},
    ],
)
async def test_bad_sources_are_refused(
    client: httpx.AsyncClient, deps: AdminDeps, fields: dict[str, str]
) -> None:
    response = await source(client, **fields)

    assert response.status_code == 422
    assert await deps.settings_store.get(SOURCES_KEY) is None


async def test_duplicates_and_too_many_are_refused(
    client: httpx.AsyncClient, deps: AdminDeps
) -> None:
    first = deps.config.sources.sources[0].id
    assert (await source(client, source_id=first)).status_code == 422
    assert (
        await client.post(
            "/admin/sources/remove",
            data={"source_id": "nope"},
            headers={"X-CSRF-Token": await token(client)},
            auth=GOOD,
        )
    ).status_code == 404
    extra = [
        {
            "id": f"s{i}",
            "name": "S",
            "site": f"s{i}.org",
            "language": "en",
            "api": False,
            "feeds": [],
        }
        for i in range(60)
    ]
    await deps.settings_store.set(SOURCES_KEY, extra)
    assert await effective_sources(deps.settings_store, deps.config.sources) is deps.config.sources
    await deps.settings_store.set(SOURCES_KEY, extra[:50])
    assert (await source(client)).status_code == 422


async def test_a_source_name_is_escaped(client: httpx.AsyncClient) -> None:
    await source(client, name="<script>alert(1)</script>")

    page = (await client.get("/admin/sources", auth=GOOD)).text

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


async def test_ingest_reads_the_saved_list_at_each_run(
    deps: AdminDeps, app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    async def fake_run(ingest_deps: IngestDeps, _options: object) -> list[Any]:
        seen.append([s.id for s in ingest_deps.catalogue.sources])
        return []

    monkeypatch.setattr(cron, "run_ingest", fake_run)
    ingest = IngestDeps(app_config.sources, app_config.thresholds.ingest, None, None, None, None)  # type: ignore[arg-type]
    ctx = {cron.INGEST_KEY: ingest, cron.SETTINGS_STORE_KEY: deps.settings_store}

    await cron.ingest_factchecks(ctx)
    only = {"id": "only", "name": "Only", "site": "only.org", "language": "en"}
    await deps.settings_store.set(SOURCES_KEY, [only])
    await cron.ingest_factchecks(ctx)

    assert seen[0] == [s.id for s in app_config.sources.sources]
    assert seen[1] == ["only"]


# --- metrics endpoint ----------------------------------------------------------------------


async def test_metrics_endpoint_serves_the_shared_counters(
    client: httpx.AsyncClient, redis: ArqRedis
) -> None:
    jm = JobMetrics()
    jm.cache_hits = ["verdict"]
    await Metrics(redis).flush(jm)

    response = await client.get("/metrics", auth=GOOD)

    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert 'cache_hits_total{layer="verdict"} 1.0' in response.text


async def test_metrics_endpoint_reports_a_redis_failure(
    client: httpx.AsyncClient, deps: AdminDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(_key: str) -> dict[str, str]:
        msg = "down"
        raise ConnectionError(msg)

    monkeypatch.setattr(deps.redis, "hgetall", broken)

    assert (await client.get("/metrics", auth=GOOD)).status_code == 503


# --- guess throttle ------------------------------------------------------------------------


async def test_repeated_wrong_passwords_lock_the_address_out(client: httpx.AsyncClient) -> None:
    bad = httpx.BasicAuth("admin", "nope")
    for attempt in range(MAX_FAILURES):
        # A made-up forwarding header must not buy a fresh allowance.
        headers = {"X-Forwarded-For": f"10.0.0.{attempt}"}
        assert (await client.get("/admin/", auth=bad, headers=headers)).status_code == 401

    locked = await client.get("/admin/", auth=GOOD, headers={"X-Forwarded-For": "10.9.9.9"})

    assert locked.status_code == 429
    assert locked.headers["retry-after"] == "300"
    assert (await client.get("/metrics", auth=GOOD)).status_code == 429


async def test_asking_for_a_login_prompt_is_not_a_failed_guess(client: httpx.AsyncClient) -> None:
    for _ in range(MAX_FAILURES + 2):
        assert (await client.get("/admin/")).status_code == 401

    assert (await client.get("/admin/", auth=GOOD)).status_code == 200


async def test_the_lockout_ends_with_its_window(client: httpx.AsyncClient, redis: ArqRedis) -> None:
    for _ in range(MAX_FAILURES):
        await client.get("/admin/", auth=httpx.BasicAuth("admin", "nope"))
    assert (await client.get("/admin/", auth=GOOD)).status_code == 429

    (key,) = [k async for k in redis.scan_iter("admin:authfail:*")]
    assert 0 < await redis.ttl(key) <= FAILURE_WINDOW_S
    await redis.delete(key)

    assert (await client.get("/admin/", auth=GOOD)).status_code == 200
