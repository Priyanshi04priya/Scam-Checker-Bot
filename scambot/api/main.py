"""HTTP application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from scambot.api.intake import Intake, open_telegram_intake
from scambot.api.webhooks import router as webhook_router
from scambot.core.logging import configure_logging
from scambot.core.settings import Settings, get_settings


def create_app(settings: Settings | None = None, telegram_intake: Intake | None = None) -> FastAPI:
    """Build the API. Tests pass ``telegram_intake``; otherwise it is opened at startup."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        if telegram_intake is not None:
            yield
            return
        async with open_telegram_intake(settings) as (intake, _bot):
            app.state.telegram_intake = intake
            yield

    app = FastAPI(
        title="Scam Checker Bot", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.settings = settings
    app.state.telegram_intake = telegram_intake
    app.include_router(webhook_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
