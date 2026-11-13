"""Intake: parse, de-duplicate, answer commands, acknowledge and enqueue."""

import enum
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol

from aiogram import Bot
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.agent.details import render_details, render_history
from scambot.api.gate import Admission, Gate
from scambot.channels.base import Button, ChannelAdapter
from scambot.channels.telegram import TelegramAdapter, create_bot
from scambot.channels.whatsapp import WhatsAppAdapter, whatsapp_client
from scambot.core.config import AppConfig, load_config
from scambot.core.contracts import Callback, Command, Job, Rejection, RejectReason
from scambot.core.db import create_engine, create_session_factory
from scambot.core.locales import LANGUAGE_NAMES, Locales, LocaleTexts, load_locales
from scambot.core.logging import get_logger
from scambot.core.media import declared_problem
from scambot.core.redis import (
    Claim,
    claim_update,
    clear_user_keys,
    create_redis,
    enqueue,
    finish_update,
    release_update,
)
from scambot.core.runtime import RuntimeConfig, RuntimeDefaults, SqlSettingsStore
from scambot.core.settings import Settings
from scambot.core.store import DetailsData, HistoryItem, SqlRequestStore
from scambot.core.tracing import trace_context

_log = get_logger(component="intake")


DELETE_CONFIRM_TTL_S = 120


class Outcome(enum.StrEnum):
    """What intake did with one update."""

    IGNORED = "ignored"
    DUPLICATE = "duplicate"
    IN_PROGRESS = "in_progress"
    COMMAND = "command"
    ENQUEUED = "enqueued"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    RATE_LIMITED = "rate_limited"
    CALLBACK = "callback"


class UserData(Protocol):
    async def delete_user(self, user_hash: str) -> int:
        """Delete everything stored for the user; returns the number of requests removed."""
        ...

    async def get_language(self, user_hash: str) -> str | None:
        """The reply language the user chose with /language, if any."""
        ...

    async def set_language(self, user_hash: str, language: str) -> None: ...

    async def history(self, user_hash: str, limit: int) -> list[HistoryItem]:
        """The user's newest requests first, with the verdict where there is one."""
        ...

    async def load_details(self, request_id: uuid.UUID, user_hash: str) -> DetailsData | None:
        """The checks behind a verdict; None when the request is unknown or not the user's."""
        ...

    async def save_feedback(self, request_id: uuid.UUID, user_hash: str, helpful: bool) -> bool:
        """Record or change the user's vote; False when the request is unknown or not theirs."""
        ...


def _update_id(parsed: Job | Command | Rejection | Callback) -> str:
    if isinstance(parsed, Job):
        return parsed.job_id
    if isinstance(parsed, Command):
        return parsed.command_id
    if isinstance(parsed, Callback):
        return parsed.callback_id
    return parsed.rejection_id


class Intake:
    """Handles every incoming update of one channel, from webhook or polling alike."""

    def __init__(  # noqa: PLR0913
        self,
        adapter: ChannelAdapter,
        redis: ArqRedis,
        locales: Locales,
        *,
        gate: Gate | None = None,
        users: UserData | None = None,
        history_limit: int = 10,
    ) -> None:
        self._adapter = adapter
        self._redis = redis
        self._locales = locales
        self._gate = gate
        self._users = users
        self._history_limit = history_limit

    async def handle(self, raw: Mapping[str, Any]) -> Outcome:
        """Process one raw platform update.

        Every update is claimed in Redis first, so a redelivered update (webhook retry,
        polling restart) is dropped and never produces a second reply. If acknowledging or
        enqueueing fails, the claim is released and the error propagates, so the platform's
        retry gets a fresh attempt instead of being dropped. ``IN_PROGRESS`` means another
        attempt holds the claim right now; the caller should ask for a redelivery later.
        """
        with trace_context() as trace_id:
            parsed = self._adapter.parse_update(raw, trace_id=trace_id)
            if parsed is None:
                _log.info("update_ignored", channel=self._adapter.channel)
                return Outcome.IGNORED
            update_id = _update_id(parsed)
            claim = await claim_update(self._redis, update_id)
            if claim is not Claim.NEW:
                _log.info("update_duplicate", update_id=update_id, state=claim.value)
                return Outcome.DUPLICATE if claim is Claim.DONE else Outcome.IN_PROGRESS
            try:
                stored = await self._stored_language(parsed.user_hash)
                if isinstance(parsed, Command):
                    await self._answer(parsed, stored)
                    outcome = Outcome.COMMAND
                elif isinstance(parsed, Rejection):
                    await self._reject(
                        parsed.chat_id, stored or parsed.preferred_lang, parsed.reason, update_id
                    )
                    outcome = Outcome.REJECTED
                elif isinstance(parsed, Callback):
                    await self._callback(parsed, stored)
                    outcome = Outcome.CALLBACK
                else:
                    outcome = await self._job(
                        parsed.model_copy(update={"user_lang": stored}), update_id
                    )
            except BaseException:
                await self._release(update_id)
                raise
            await self._finish(update_id)
            return outcome

    async def _stored_language(self, user_hash: str) -> str | None:
        """The language the user chose, or None; a failure only means no choice is known."""
        if self._users is None:
            return None
        try:
            return await self._users.get_language(user_hash) or None
        except Exception as exc:
            _log.warning("language_lookup_failed", error=type(exc).__name__)
            return None

    async def _job(self, job: Job, update_id: str) -> Outcome:
        """Admit, validate and enqueue one job."""
        if self._gate is not None:
            decision = await self._gate.admit(job.user_hash)
            if decision.admission is Admission.BLOCKED:
                return Outcome.BLOCKED
            if decision.admission is Admission.LIMITED:
                text = self._locales.get(job.user_lang or job.preferred_lang).rate_limited.format(
                    limit=decision.limit, minutes=decision.minutes
                )
                await self._adapter.send_text(job.chat_id, text)
                return Outcome.RATE_LIMITED
        if problem := declared_problem(job.kind, job.media):
            await self._refund(job)
            await self._reject(job.chat_id, job.user_lang or job.preferred_lang, problem, update_id)
            return Outcome.REJECTED
        try:
            await self._accept(job)
        except BaseException:
            await self._refund(job)
            raise
        return Outcome.ENQUEUED

    async def _refund(self, job: Job) -> None:
        if self._gate is not None:
            await self._gate.refund(job.user_hash)

    async def _release(self, update_id: str) -> None:
        # Called while another error propagates; that error is the one worth reporting. If
        # Redis is unreachable the short claim expires by itself.
        try:
            await release_update(self._redis, update_id)
        except Exception as exc:
            _log.warning("claim_release_failed", update_id=update_id, error=type(exc).__name__)

    async def _finish(self, update_id: str) -> None:
        # The reply or job is already out, so failing here would only make the platform
        # redeliver it. The short claim still drops redeliveries until it expires.
        try:
            await finish_update(self._redis, update_id)
        except Exception as exc:
            _log.warning("claim_finish_failed", update_id=update_id, error=type(exc).__name__)

    async def _answer(self, command: Command, stored: str | None) -> None:
        lang = stored or command.preferred_lang
        texts = self._locales.get(lang).commands
        users = self._users
        if users is not None and command.name == "delete":
            if command.channel == "whatsapp" and not await self._confirmed(command):
                await self._adapter.send_text(command.chat_id, texts.delete_confirm)
                return
            await self._delete(users, command, texts.delete, texts.delete_failed)
            return
        if users is not None and command.name == "language":
            await self._language_command(users, command, lang)
            return
        if users is not None and command.name == "history":
            await self._history(users, command, lang)
            return
        known = {"start": texts.start, "help": texts.help, "privacy": texts.privacy}
        await self._adapter.send_text(command.chat_id, known.get(command.name, texts.unknown))
        _log.info(
            "command_answered",
            update_id=command.command_id,
            command=command.name if command.name in known else "unknown",
            language=self._locales.resolve(lang),
        )

    async def _language_command(self, users: UserData, command: Command, lang: str | None) -> None:
        texts = self._locales.get(lang)
        if not command.args:
            row = [Button(LANGUAGE_NAMES.get(c, c), f"l:{c}") for c in self._locales.selectable()]
            await self._adapter.send_text(command.chat_id, texts.language.prompt, buttons=[row])
            return
        code = self._locales.normalise_choice(command.args)
        if code is None:
            await self._adapter.send_text(command.chat_id, texts.language.invalid)
            return
        await self._adapter.send_text(
            command.chat_id, await self._set_language(users, command.user_hash, code, texts)
        )

    async def _set_language(
        self, users: UserData, user_hash: str, code: str, current: LocaleTexts
    ) -> str:
        """Store the choice; the answer is written in the newly chosen language."""
        try:
            await users.set_language(user_hash, code)
        except Exception as exc:
            _log.error("language_save_failed", error=type(exc).__name__)
            return current.feedback.unavailable
        _log.info("language_set", language=code)
        return self._locales.get(code).language.saved.format(
            language=LANGUAGE_NAMES.get(code, code)
        )

    async def _history(self, users: UserData, command: Command, lang: str | None) -> None:
        texts = self._locales.get(lang)
        try:
            items = await users.history(command.user_hash, self._history_limit)
        except Exception as exc:
            _log.error("history_failed", error=type(exc).__name__)
            await self._adapter.send_text(command.chat_id, texts.feedback.unavailable)
            return
        await self._adapter.send_text(command.chat_id, render_history(items, texts))

    async def _callback(self, cb: Callback, stored: str | None) -> None:
        """Handle a button press; the press is always answered, even when handling fails."""
        lang = stored or cb.preferred_lang
        texts = self._locales.get(lang)
        toast: str | None = None
        try:
            if self._gate is not None and await self._gate.is_blocked(cb.user_hash):
                _log.info("callback_blocked")
            elif self._users is not None:
                toast = await self._press(self._users, cb, stored, texts)
        except Exception as exc:  # a button press is not worth a redelivery loop
            _log.error("callback_failed", action=cb.action, error=type(exc).__name__)
            toast = texts.feedback.unavailable
        finally:
            try:
                await self._adapter.answer_callback(cb, toast)
            except Exception as exc:
                _log.warning("callback_answer_failed", error=type(exc).__name__)

    async def _press(  # noqa: PLR0911
        self, users: UserData, cb: Callback, stored: str | None, texts: LocaleTexts
    ) -> str | None:
        """Carry out one button press; returns the short notice to show, if any."""
        if cb.action == "expired":
            return texts.details.expired
        if cb.action == "language":
            if self._locales.normalise_choice(cb.value) != cb.value:
                _log.info("callback_ignored", reason="language")
                return None
            await self._adapter.send_text(
                cb.chat_id, await self._set_language(users, cb.user_hash, cb.value, texts)
            )
            return None
        try:
            request_id = uuid.UUID(cb.value)
        except ValueError:
            return texts.feedback.unavailable
        if cb.action == "details":
            data = await users.load_details(request_id, cb.user_hash)
            if data is None:
                return texts.feedback.unavailable
            shown = self._locales.get(stored or data.reply_language)
            await self._adapter.send_text(cb.chat_id, render_details(data, shown))
            return None
        helpful = cb.action == "helpful"
        if not await users.save_feedback(request_id, cb.user_hash, helpful):
            return texts.feedback.unavailable
        return texts.feedback.thanks_helpful if helpful else texts.feedback.thanks_wrong

    async def _confirmed(self, command: Command) -> bool:
        """WhatsApp has no explicit command: a plain "delete" must be repeated to take effect."""
        key = f"delete_confirm:{command.user_hash}"
        try:
            if await self._redis.get(key) is not None:
                await self._redis.delete(key)
                return True
            await self._redis.set(key, "1", ex=DELETE_CONFIRM_TTL_S)
        except Exception as exc:  # fail safe: without the flag nothing is erased
            _log.warning("delete_confirm_failed", error=type(exc).__name__)
        return False

    async def _delete(self, users: UserData, command: Command, done: str, failed: str) -> None:
        try:
            removed = await users.delete_user(command.user_hash)
            await clear_user_keys(self._redis, command.user_hash)
        except Exception as exc:  # never claim success when the records may still exist
            _log.error("user_delete_failed", error=type(exc).__name__)
            await self._adapter.send_text(command.chat_id, failed)
            return
        _log.info("user_deleted", requests=removed)
        await self._adapter.send_text(command.chat_id, done.format(count=removed))

    async def _reject(
        self, chat_id: str, language: str | None, reason: RejectReason, update_id: str
    ) -> None:
        await self._adapter.send_text(
            chat_id, getattr(self._locales.get(language).rejected, reason)
        )
        _log.info("media_rejected", update_id=update_id, reason=reason, stage="intake")

    async def _accept(self, job: Job) -> None:
        checking = self._locales.get(job.user_lang or job.preferred_lang).checking
        ack_message_id = await self._adapter.send_ack(job, checking)
        job = job.model_copy(update={"ack_message_id": ack_message_id})
        queued = await enqueue(self._redis, job)
        _log.info(
            "job_enqueued" if queued else "job_already_queued",
            job_id=job.job_id,
            kind=job.kind,
            user_hash=job.user_hash,
        )


def build_intake(  # noqa: PLR0913
    adapter: ChannelAdapter,
    *,
    redis: ArqRedis,
    sessions: async_sessionmaker[AsyncSession],
    runtime: RuntimeConfig,
    config: AppConfig,
    locales: Locales,
) -> Intake:
    """An intake for ``adapter`` with the rate limit, user data and history settings applied."""
    return Intake(
        adapter,
        redis,
        locales,
        gate=Gate(runtime, redis, config.thresholds.limits.rate_window_s),
        users=SqlRequestStore(sessions),
        history_limit=config.thresholds.history.limit,
    )


@asynccontextmanager
async def open_telegram_intake(
    settings: Settings,
    *,
    redis: ArqRedis | None = None,
    sessions: async_sessionmaker[AsyncSession] | None = None,
    runtime: RuntimeConfig | None = None,
) -> AsyncIterator[tuple[Intake, Bot]]:
    """Build the Telegram intake from settings and close its connections on exit.

    Connections passed in (the API shares one set between intake and admin pages) belong to
    the caller and are left open.
    """
    config = load_config()
    bot = create_bot(settings.telegram_bot_token, settings.telegram_api_url)
    own_redis = redis is None
    if redis is None:
        redis = await create_redis(settings.redis_url)
    engine = None
    if sessions is None:
        engine = create_engine(settings.database_url)
        sessions = create_session_factory(engine)
    if runtime is None:
        runtime = RuntimeConfig(
            SqlSettingsStore(sessions),
            redis,
            RuntimeDefaults.build(config.thresholds, settings),
            config.thresholds.limits.config_cache_s,
        )
    intake = build_intake(
        TelegramAdapter(bot, settings.user_id_pepper),
        redis=redis,
        sessions=sessions,
        runtime=runtime,
        config=config,
        locales=load_locales(),
    )
    try:
        yield intake, bot
    finally:
        if engine is not None:
            await engine.dispose()
        if own_redis:
            await redis.aclose()
        await bot.session.close()


@asynccontextmanager
async def open_whatsapp_intake(
    settings: Settings,
    *,
    redis: ArqRedis,
    sessions: async_sessionmaker[AsyncSession],
    runtime: RuntimeConfig,
) -> AsyncIterator[Intake]:
    """Build the WhatsApp intake; the connections passed in belong to the caller."""
    locales = load_locales()
    async with whatsapp_client() as client:
        adapter = WhatsAppAdapter(
            client,
            token=settings.whatsapp_token,
            phone_number_id=settings.whatsapp_phone_number_id,
            api_version=settings.whatsapp_api_version,
            graph_url=settings.whatsapp_graph_url,
            allowlist=settings.whatsapp_allowlist,
            user_id_pepper=settings.user_id_pepper,
            mark_read=settings.whatsapp_mark_read,
            locales=locales,
        )
        yield build_intake(
            adapter,
            redis=redis,
            sessions=sessions,
            runtime=runtime,
            config=load_config(),
            locales=locales,
        )
