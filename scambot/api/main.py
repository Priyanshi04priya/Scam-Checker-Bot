"""HTTP application factory."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from scambot.api.admin.auth import AdminAuth
from scambot.api.admin.deps import AdminDeps
from scambot.api.admin.queries import StatsStore
from scambot.api.admin.routes import SECURITY_HEADERS, STATIC_DIR, metrics_router
from scambot.api.admin.routes import router as admin_router
from scambot.api.intake import Intake, open_telegram_intake, open_whatsapp_intake
from scambot.api.webhooks import router as webhook_router
from scambot.core.config import load_config
from scambot.core.db import create_engine, create_session_factory
from scambot.core.logging import configure_logging, get_logger
from scambot.core.redis import create_redis
from scambot.core.runtime import RuntimeConfig, RuntimeDefaults, SqlSettingsStore
from scambot.core.settings import Settings, get_settings

_log = get_logger(component="api")


def create_app(
    settings: Settings | None = None,
    telegram_intake: Intake | None = None,
    admin: AdminDeps | None = None,
    whatsapp_intake: Intake | None = None,
) -> FastAPI:
    """Build the API. Tests pass ``telegram_intake`` and ``admin``; otherwise both are opened
    at startup and share one database engine and one Redis pool."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        if settings.environment == "prod" and not settings.admin_password_hash.get_secret_value():
            _log.info("admin_disabled")
        if settings.whatsapp_enabled and not settings.whatsapp_allowlist:
            _log.warning("whatsapp_allowlist_empty")
        elif not settings.whatsapp_enabled:
            _log.info("whatsapp_disabled")
        if telegram_intake is not None or whatsapp_intake is not None:
            yield
            return
        config = load_config()
        redis = await create_redis(settings.redis_url)
        engine = create_engine(settings.database_url)
        try:
            sessions = create_session_factory(engine)
            store = SqlSettingsStore(sessions)
            runtime = RuntimeConfig(
                store,
                redis,
                RuntimeDefaults.build(config.thresholds, settings),
                config.thresholds.limits.config_cache_s,
            )
            app.state.admin = AdminDeps(StatsStore(sessions), store, runtime, redis, config)
            async with AsyncExitStack() as stack:
                intake, _bot = await stack.enter_async_context(
                    open_telegram_intake(settings, redis=redis, sessions=sessions, runtime=runtime)
                )
                app.state.telegram_intake = intake
                if settings.whatsapp_enabled:
                    app.state.whatsapp_intake = await stack.enter_async_context(
                        open_whatsapp_intake(
                            settings, redis=redis, sessions=sessions, runtime=runtime
                        )
                    )
                yield
        finally:
            await engine.dispose()
            await redis.aclose()

    app = FastAPI(
        title="Scam Checker Bot", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.settings = settings
    app.state.telegram_intake = telegram_intake
    app.state.whatsapp_intake = whatsapp_intake
    app.state.admin = admin
    app.include_router(webhook_router)

    password_hash = settings.admin_password_hash.get_secret_value()
    if password_hash:
        limits = (admin.config if admin else load_config()).thresholds.admin
        app.state.admin_auth = AdminAuth(
            settings.admin_user,
            password_hash,
            settings.user_id_pepper.get_secret_value(),
            cache_s=limits.credentials_cache_s,
            csrf_ttl_s=limits.csrf_ttl_s,
        )

        @app.middleware("http")
        async def secure_headers(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            response = await call_next(request)
            path = request.url.path
            if path == "/metrics" or path.startswith("/admin"):
                response.headers.update(SECURITY_HEADERS)
            return response

        app.include_router(admin_router)
        app.include_router(metrics_router)
        app.mount("/admin/static", StaticFiles(directory=STATIC_DIR), name="admin-static")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
