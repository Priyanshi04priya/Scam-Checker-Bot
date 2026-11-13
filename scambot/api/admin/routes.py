"""Admin pages (server-rendered, refreshed with htmx) and the metrics endpoint."""

from pathlib import Path
from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends, Form, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from scambot.api.admin import controls
from scambot.api.admin.auth import AdminAuth, require_admin, require_csrf
from scambot.api.admin.controls import ControlError
from scambot.api.admin.deps import AdminDeps
from scambot.core import metrics

HERE: Final = Path(__file__).parent
STATIC_DIR: Final = HERE / "static"
SECURITY_HEADERS: Final = {
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
}

_env = Environment(
    loader=FileSystemLoader(HERE / "templates"),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])
metrics_router = APIRouter(dependencies=[Depends(require_admin)])


def _deps(request: Request) -> AdminDeps:
    deps: AdminDeps = request.app.state.admin
    return deps


def _render(request: Request, name: str, status: int = 200, **context: Any) -> HTMLResponse:
    auth: AdminAuth = request.app.state.admin_auth
    now = int(_deps(request).now().timestamp())
    html = _env.get_template(name).render(csrf=auth.make_token(now), **context)
    return HTMLResponse(html, status_code=status)


def _days(request: Request, days: int | None) -> int:
    limits = _deps(request).config.thresholds.admin
    return min(max(days if days is not None else limits.days_default, 1), limits.days_max)


async def _stats_context(request: Request, days: int | None) -> dict[str, Any]:
    deps = _deps(request)
    shown = _days(request, days)
    overview = await deps.stats.overview(shown, deps.now())
    return {
        "days": shown,
        "o": overview,
        "budget": deps.config.thresholds.costs.budget_per_check_usd,
        "max_day": max((count for _day, count in overview.per_day), default=0) or 1,
    }


@router.get("/", response_class=HTMLResponse)
async def overview(request: Request, days: Annotated[int | None, Query()] = None) -> HTMLResponse:
    return _render(request, "overview.html", days=_days(request, days), page="overview")


@router.get("/fragments/stats", response_class=HTMLResponse)
async def stats_fragment(
    request: Request, days: Annotated[int | None, Query()] = None
) -> HTMLResponse:
    return _render(request, "_stats.html", **await _stats_context(request, days))


async def _controls_context(request: Request) -> dict[str, Any]:
    deps = _deps(request)
    return {
        "limits": await deps.runtime.describe(),
        "users": await deps.stats.recent_users(
            deps.config.thresholds.admin.recent_users, deps.now()
        ),
        "blocked": await deps.stats.blocked_users(),
    }


async def _sources_context(request: Request) -> dict[str, Any]:
    sources, overridden = await controls.current_sources(_deps(request))
    return {"sources": sources.sources, "overridden": overridden}


async def _respond(
    request: Request,
    page: str,
    context: dict[str, Any],
    *,
    error: str | None = None,
    status: int = 200,
) -> Response:
    """The refreshed fragment for htmx, a redirect for a plain form, or the page with the error."""
    htmx = request.headers.get("hx-request") == "true"
    if error is None and not htmx:
        return RedirectResponse(f"/admin/{page}", status_code=303)
    template = f"_{page}.html" if htmx else f"{page}.html"
    return _render(request, template, status, error=error, page=page, **context)


@router.get("/controls", response_class=HTMLResponse)
async def controls_page(request: Request) -> HTMLResponse:
    return _render(
        request, "controls.html", page="controls", error=None, **await _controls_context(request)
    )


@router.post("/users/block", dependencies=[Depends(require_csrf)])
async def block_user(
    request: Request, prefix: Annotated[str, Form()], blocked: Annotated[str, Form()] = "true"
) -> Response:
    error, status = None, 200
    try:
        await controls.set_blocked(_deps(request), prefix, blocked.lower() == "true")
    except ControlError as exc:
        error, status = exc.message, exc.status
    return await _respond(
        request, "controls", await _controls_context(request), error=error, status=status
    )


@router.post("/settings", dependencies=[Depends(require_csrf)])
async def change_settings(
    request: Request,
    rate_limit_per_hour: Annotated[str, Form()] = "",
    cache_window_days: Annotated[str, Form()] = "",
) -> Response:
    error, status = None, 200
    deps = _deps(request)
    try:
        # Check both values before writing either.
        for name, raw in (
            ("rate_limit_per_hour", rate_limit_per_hour),
            ("cache_window_days", cache_window_days),
        ):
            controls.check_limit(name, raw)
        await controls.set_limit(deps, "rate_limit_per_hour", rate_limit_per_hour)
        await controls.set_limit(deps, "cache_window_days", cache_window_days)
    except ControlError as exc:
        error, status = exc.message, exc.status
    return await _respond(
        request, "controls", await _controls_context(request), error=error, status=status
    )


@router.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request) -> HTMLResponse:
    return _render(
        request, "sources.html", page="sources", error=None, **await _sources_context(request)
    )


async def _source_action(request: Request, action: Any) -> Response:
    error, status = None, 200
    try:
        await action
    except ControlError as exc:
        error, status = exc.message, exc.status
    return await _respond(
        request, "sources", await _sources_context(request), error=error, status=status
    )


@router.post("/sources/add", dependencies=[Depends(require_csrf)])
async def add_source(  # noqa: PLR0913, PLR0917
    request: Request,
    source_id: Annotated[str, Form()],
    name: Annotated[str, Form()],
    site: Annotated[str, Form()],
    language: Annotated[str, Form()],
    api: Annotated[str | None, Form()] = None,
    feeds: Annotated[str, Form()] = "",
) -> Response:
    return await _source_action(
        request,
        controls.add_source(
            _deps(request),
            source_id=source_id,
            name=name,
            site=site,
            language=language,
            api=api is not None,
            feeds=feeds,
        ),
    )


@router.post("/sources/remove", dependencies=[Depends(require_csrf)])
async def remove_source(request: Request, source_id: Annotated[str, Form()]) -> Response:
    return await _source_action(request, controls.remove_source(_deps(request), source_id))


@router.post("/sources/reset", dependencies=[Depends(require_csrf)])
async def reset_sources(request: Request) -> Response:
    return await _source_action(request, controls.reset_sources(_deps(request)))


@metrics_router.get("/metrics")
async def prometheus(request: Request) -> Response:
    try:
        body = await metrics.render(_deps(request).redis)
    except Exception:
        return Response("metrics unavailable", status_code=503)
    return Response(body, media_type=metrics.CONTENT_TYPE)
