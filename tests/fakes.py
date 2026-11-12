"""In-memory test doubles shared by unit and integration tests."""

import asyncio
import datetime as dt
import itertools
import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, SecretStr

from scambot.agent.executor import ToolSession
from scambot.core.contracts import Channel, Command, ExecutedCheck, Job, Verdict
from scambot.core.llm import LLMError, LLMImage, LLMUsage, PromptLibrary, record_usage
from scambot.core.security import user_hash
from scambot.core.settings import DEV_USER_ID_PEPPER
from scambot.core.store import NewRequest, RequestStatus, StartedRequest, StoredResult

_COMMAND_RE = re.compile(r"/([A-Za-z0-9_]{1,32})(?:\s|$)")


def fake_update(
    update_id: int, text: str, *, user_id: int = 7, lang: str | None = "en"
) -> dict[str, Any]:
    """A raw update in the fake channel's own minimal format."""
    return {"update_id": update_id, "user_id": user_id, "text": text, "lang": lang}


@dataclass
class FakeAdapter:
    """Channel adapter that keeps every sent message in memory and supports editing."""

    channel: Channel = "telegram"
    messages: dict[str, str] = field(default_factory=dict)
    log: list[tuple[str, str, str]] = field(default_factory=list)
    fail_ack: bool = False
    _ids: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    def parse_update(self, raw: Mapping[str, Any], *, trace_id: str) -> Job | Command | None:
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            return None
        update_id = f"{self.channel}:{raw['update_id']}"
        chat_id = str(raw["user_id"])
        hashed = user_hash(raw["user_id"], SecretStr(DEV_USER_ID_PEPPER))
        # Same rule as the Telegram adapter: a name must follow the "/" directly.
        if match := _COMMAND_RE.match(text):
            return Command(
                command_id=update_id,
                trace_id=trace_id,
                channel=self.channel,
                chat_id=chat_id,
                user_hash=hashed,
                name=match.group(1).lower(),
                preferred_lang=raw.get("lang"),
            )
        return Job(
            job_id=update_id,
            trace_id=trace_id,
            channel=self.channel,
            chat_id=chat_id,
            user_hash=hashed,
            kind="text",
            text=text,
            preferred_lang=raw.get("lang"),
            received_at=datetime.now(UTC),
        )

    def _new_message(self, action: str, chat_id: str, text: str) -> str:
        message_id = str(next(self._ids))
        self.messages[message_id] = text
        self.log.append((action, chat_id, text))
        return message_id

    async def send_ack(self, job: Job, text: str) -> str:
        if self.fail_ack:
            msg = "platform unavailable"
            raise ConnectionError(msg)
        return self._new_message("ack", job.chat_id, text)

    async def send_verdict(self, job: Job, text: str) -> None:
        if job.ack_message_id in self.messages:
            self.messages[job.ack_message_id] = text
            self.log.append(("edit", job.chat_id, text))
        else:
            self._new_message("verdict", job.chat_id, text)

    async def send_text(self, chat_id: str, text: str) -> None:
        self._new_message("text", chat_id, text)


FakeResponse = BaseModel | Exception | Callable[[Mapping[str, str]], BaseModel]


class FakeLLMClient:
    """Answers each named prompt with a canned response, rendering through the real prompts."""

    def __init__(
        self, prompts: PromptLibrary, responses: Mapping[str, FakeResponse] | None = None
    ) -> None:
        self._prompts = prompts
        self.responses: dict[str, FakeResponse] = dict(responses or {})
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def structured[T: BaseModel](
        self,
        prompt_name: str,
        variables: Mapping[str, str],
        schema: type[T],
        images: Sequence[LLMImage] | None = None,
    ) -> T:
        self._prompts.render(prompt_name, variables)  # variable names must match the file
        self.calls.append((prompt_name, dict(variables)))
        record_usage(
            LLMUsage(call=prompt_name, model="claude-opus-5-5", input_tokens=100, output_tokens=20)
        )
        response = self.responses.get(prompt_name)
        if response is None:
            msg = f"no fake response for {prompt_name}"
            raise LLMError(msg)
        if isinstance(response, Exception):
            raise response
        if callable(response) and not isinstance(response, BaseModel):
            response = response(variables)
        return schema.model_validate(response.model_dump())

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


class SlowLLMClient:
    """An LLM that never answers in time."""

    async def structured[T: BaseModel](
        self,
        prompt_name: str,
        variables: Mapping[str, str],
        schema: type[T],
        images: Sequence[LLMImage] | None = None,
    ) -> T:
        await asyncio.sleep(10)
        raise AssertionError  # pragma: no cover


@dataclass
class FakeReputation:
    unsafe: dict[str, str] | Exception = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)

    async def unsafe_urls(self, urls: Sequence[str]) -> dict[str, str]:
        self.calls.append(list(urls))
        if isinstance(self.unsafe, Exception):
            raise self.unsafe
        return {url: threat for url, threat in self.unsafe.items() if url in urls}


@dataclass
class StoredRequest:
    request: NewRequest
    id: uuid.UUID
    status: RequestStatus = "received"
    replied_at: dt.datetime | None = None
    checks: list[ExecutedCheck] = field(default_factory=list)
    verdict: Verdict | None = None
    reply_text: str = ""
    reply_language: str = ""
    claims: list[str] = field(default_factory=list)
    language: str | None = None
    cost_usd: Any = None


@dataclass
class InMemoryStore:
    """``RequestStore`` kept in dictionaries; ``fail`` makes every method raise."""

    fail: bool = False
    requests: dict[str, StoredRequest] = field(default_factory=dict)
    users: set[str] = field(default_factory=set)

    def _check(self) -> None:
        if self.fail:
            msg = "database unavailable"
            raise ConnectionError(msg)

    def _by_id(self, request_id: uuid.UUID) -> StoredRequest:
        return next(r for r in self.requests.values() if r.id == request_id)

    async def start(self, request: NewRequest) -> StartedRequest:
        self._check()
        existing = self.requests.get(request.job_id)
        if existing is not None:
            return StartedRequest(id=existing.id, status=existing.status, created=False)
        self.users.add(request.user_hash)
        row = StoredRequest(request=request, id=uuid.uuid4())
        self.requests[request.job_id] = row
        return StartedRequest(id=row.id, status=row.status, created=True)

    async def save(self, request_id: uuid.UUID, result: StoredResult) -> None:
        self._check()
        row = self._by_id(request_id)
        row.checks = list(result.checks)
        row.verdict = result.verdict
        row.reply_text = result.reply_text
        row.reply_language = result.reply_language
        row.claims = list(result.claims)
        row.language = result.language or row.language
        row.cost_usd = result.cost_usd

    async def set_status(
        self, request_id: uuid.UUID, status: RequestStatus, *, at: dt.datetime | None = None
    ) -> None:
        self._check()
        row = self._by_id(request_id)
        row.status = status
        if status == "replied":
            row.replied_at = at or dt.datetime.now(dt.UTC)


class _RefusedSession:
    async def __aenter__(self) -> ToolSession:
        msg = "checker server unreachable"
        raise ConnectionError(msg)

    async def __aexit__(self, *_exc: object) -> None:
        return None  # pragma: no cover


class FailingToolCaller:
    """A checker server that cannot be reached."""

    def session(self) -> AbstractAsyncContextManager[ToolSession]:
        return _RefusedSession()
