"""In-memory test doubles shared by unit and integration tests."""

import asyncio
import datetime as dt
import hashlib
import itertools
import json
import math
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SecretStr

from scambot.agent.executor import ToolSession
from scambot.core.contracts import (
    Channel,
    Command,
    ExecutedCheck,
    Job,
    MediaMeta,
    Rejection,
    Verdict,
)
from scambot.core.llm import LLMError, LLMImage, LLMUsage, PromptLibrary, record_usage
from scambot.core.media import MediaUnavailable
from scambot.core.media_store import ORIGINAL, MediaStore
from scambot.core.security import user_hash
from scambot.core.settings import DEV_USER_ID_PEPPER
from scambot.core.store import NewRequest, RequestStatus, StartedRequest, StoredResult
from scambot.factcheck_index.records import ArticleRecord, ClaimReview, passage_text
from scambot.factcheck_index.search import EmbeddedArticle, FactcheckIndex
from scambot.mcp_server.checkers.caption import CaptionReply
from scambot.mcp_server.checkers.factcheck import RelevanceReply
from scambot.mcp_server.clients.pagefetch import FetchError
from scambot.mcp_server.clients.sightengine import SynthScores
from scambot.mcp_server.clients.vision import WebPage
from scambot.preprocess.base import Extracted
from tests.media import jpeg_bytes

_COMMAND_RE = re.compile(r"/([A-Za-z0-9_]{1,32})(?:\s|$)")


def fake_update(
    update_id: int, text: str, *, user_id: int = 7, lang: str | None = "en"
) -> dict[str, Any]:
    """A raw update in the fake channel's own minimal format."""
    return {"update_id": update_id, "user_id": user_id, "text": text, "lang": lang}


def fake_media_update(
    update_id: int,
    kind: str,
    ref: str,
    *,
    caption: str | None = None,
    size: int | None = None,
    mime: str | None = None,
    source: str = "photo",
    user_id: int = 7,
    lang: str | None = "en",
) -> dict[str, Any]:
    """A raw media update in the fake channel's own format."""
    media = {"kind": kind, "ref": ref, "size": size, "mime": mime, "source": source}
    return {
        "update_id": update_id,
        "user_id": user_id,
        "text": caption,
        "lang": lang,
        "media": media,
    }


@dataclass
class FakeAdapter:
    """Channel adapter that keeps every sent message in memory and supports editing."""

    channel: Channel = "telegram"
    messages: dict[str, str] = field(default_factory=dict)
    log: list[tuple[str, str, str]] = field(default_factory=list)
    fail_ack: bool = False
    files: dict[str, bytes] = field(default_factory=dict)
    fail_download: int = 0
    downloads: int = 0
    _ids: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    def parse_update(
        self, raw: Mapping[str, Any], *, trace_id: str
    ) -> Job | Command | Rejection | None:
        if "user_id" not in raw:
            return None
        update_id = f"{self.channel}:{raw['update_id']}"
        chat_id = str(raw["user_id"])
        hashed = user_hash(raw["user_id"], SecretStr(DEV_USER_ID_PEPPER))
        if "reject" in raw:
            return Rejection(
                rejection_id=update_id,
                trace_id=trace_id,
                channel=self.channel,
                chat_id=chat_id,
                user_hash=hashed,
                reason=raw["reject"],
                preferred_lang=raw.get("lang"),
            )
        if (media := raw.get("media")) is not None:
            return Job(
                job_id=update_id,
                trace_id=trace_id,
                channel=self.channel,
                chat_id=chat_id,
                user_hash=hashed,
                kind=media["kind"],
                text=raw.get("text"),
                media_ref=media["ref"],
                media=MediaMeta(
                    source=media["source"], file_size=media["size"], mime_type=media["mime"]
                ),
                preferred_lang=raw.get("lang"),
                received_at=datetime.now(UTC),
            )
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            return None
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

    async def download_media(
        self, job: Job, store: MediaStore, key: str, *, max_bytes: int, timeout_s: float
    ) -> Path:
        self.downloads += 1
        if self.fail_download > 0:
            self.fail_download -= 1
            msg = "network"
            raise MediaUnavailable(msg)
        data = self.files[job.media_ref or ""]

        async def chunks() -> AsyncIterator[bytes]:
            for start in range(0, len(data), 65536):
                yield data[start : start + 65536]

        return await store.put(key, ORIGINAL, chunks(), max_bytes=max_bytes)

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
    sha256: str | None = None
    phash: int | None = None


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
        row = StoredRequest(request=request, id=uuid.uuid4(), sha256=request.sha256)
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
        row.sha256 = result.sha256 or row.sha256
        row.phash = result.phash if result.phash is not None else row.phash

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


_WORD_RE = re.compile(r"[^\s.,;:!?()\"'।]+")


@dataclass
class KeywordEmbedder:
    """Deterministic bag-of-words embeddings; aliases map words across languages."""

    aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    dim: int = 384
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def vector(self, text: str) -> list[float]:
        counts = [0.0] * self.dim
        for word in _WORD_RE.findall(text.casefold()):
            for token in self.aliases.get(word, (word,)):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                counts[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(c * c for c in counts))
        if norm == 0:
            return [1.0] + [0.0] * (self.dim - 1)
        return [c / norm for c in counts]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(("query", list(texts)))
        return [self.vector(text) for text in texts]

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(("passage", list(texts)))
        return [self.vector(text) for text in texts]


async def seed(index: FactcheckIndex, embedder: KeywordEmbedder, *records: ArticleRecord) -> None:
    """Store ``records`` in ``index`` with their passage embeddings."""
    await index.upsert(
        [EmbeddedArticle(record, embedder.vector(passage_text(record))) for record in records]
    )


@dataclass
class FakeClaimSearch:
    reviews: list[ClaimReview] | Exception = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    async def search(self, query: str, *, page_size: int) -> list[ClaimReview]:
        self.queries.append(query)
        if isinstance(self.reviews, Exception):
            raise self.reviews
        return list(self.reviews)


@dataclass
class FakePublisherReviews:
    by_site: dict[str, list[ClaimReview] | Exception] = field(default_factory=dict)
    calls: list[tuple[str, int, int, int]] = field(default_factory=list)

    async def by_publisher(
        self, site: str, *, max_age_days: int, page_size: int, max_pages: int
    ) -> list[ClaimReview]:
        self.calls.append((site, max_age_days, page_size, max_pages))
        reviews = self.by_site.get(site, [])
        if isinstance(reviews, Exception):
            raise reviews
        return list(reviews)


class FailingIndex:
    """A fact-check index whose database is down."""

    @staticmethod
    def _fail() -> ConnectionError:
        return ConnectionError("index unavailable")

    async def nearest(self, embedding: Sequence[float], k: int) -> list[ArticleRecord]:
        raise self._fail()

    async def full_text(self, query: str, k: int) -> list[ArticleRecord]:
        raise self._fail()

    async def upsert(self, items: Sequence[EmbeddedArticle]) -> int:
        raise self._fail()

    async def known_urls(self, urls: Sequence[str]) -> set[str]:
        raise self._fail()


@dataclass
class _Rows:
    rows: list[list[float]]

    def tolist(self) -> list[list[float]]:
        return self.rows


@dataclass
class StubEncoder:
    """Stands in for a sentence-transformers model."""

    dim: int = 384
    calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)

    def encode(self, inputs: list[str], **kwargs: Any) -> _Rows:
        self.calls.append((list(inputs), kwargs))
        return _Rows([[float(i)] + [0.0] * (self.dim - 1) for i in range(len(inputs))])


def judge(
    relation: str = "same", polarity: str = "false", *, about: str = "laptop"
) -> Callable[[Mapping[str, str]], RelevanceReply]:
    """Answer ``relation``/``polarity`` for candidates whose title mentions ``about``."""

    def answer(variables: Mapping[str, str]) -> RelevanceReply:
        candidates = json.loads(variables["candidates_json"])
        return RelevanceReply.model_validate(
            {
                "judgments": [
                    {
                        "candidate": c["candidate"],
                        "relation": relation if about in c["title"].lower() else "different",
                        "polarity": polarity,
                    }
                    for c in candidates
                ]
            }
        )

    return answer


@dataclass
class FakeOcr:
    """Reads a fixed text from any image."""

    text: str = ""
    calls: list[Path] = field(default_factory=list)

    def extract(self, path: Path) -> Extracted:
        self.calls.append(path)
        return Extracted(self.text, None)


@dataclass
class FakeStt:
    text: str = ""
    language: str | None = None

    def extract(self, path: Path) -> Extracted:
        return Extracted(self.text, self.language)


@dataclass
class FakeKeyframes:
    """Writes ``count`` small distinct JPEGs next to the source file."""

    count: int = 0

    def extract(self, path: Path, max_frames: int) -> list[Path]:
        frames: list[Path] = []
        for i in range(min(self.count, max_frames)):
            target = path.with_name(f"kf_{i:02d}.jpg")
            target.write_bytes(jpeg_bytes((40 * i % 256, 90, 255 - 40 * i % 256)))
            frames.append(target)
        return frames


@dataclass
class SlowEngine:
    """An engine (any kind) that takes longer than its time limit."""

    seconds: float = 1.0

    def extract(self, path: Path, max_frames: int = 0) -> Any:
        time.sleep(self.seconds)
        return Extracted("late", None)


@dataclass
class BrokenEngine:
    error: Exception

    def extract(self, path: Path, max_frames: int = 0) -> Any:
        raise self.error


@dataclass
class FakeWebSearch:
    pages: list[WebPage] | Exception = field(default_factory=list)
    calls: list[tuple[int, int]] = field(default_factory=list)

    async def pages_with_matches(self, image: bytes, *, max_results: int) -> list[WebPage]:
        self.calls.append((len(image), max_results))
        if isinstance(self.pages, Exception):
            raise self.pages
        return list(self.pages)


@dataclass
class FakeImageAnalyzer:
    by_name: dict[str, SynthScores | Exception] = field(default_factory=dict)
    default: SynthScores = field(default_factory=lambda: SynthScores(0.02, 0.01))
    calls: list[tuple[str, int]] = field(default_factory=list)

    async def scores(self, image: bytes, filename: str) -> SynthScores:
        self.calls.append((filename, len(image)))
        result = self.by_name.get(filename, self.default)
        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class FakePageFetcher:
    pages: dict[str, str | FetchError] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    async def get_html(self, url: str) -> str:
        self.calls.append(url)
        page = self.pages.get(url, FetchError("status_404"))
        if isinstance(page, FetchError):
            raise page
        return page


@dataclass
class StaticClock:
    day: dt.date

    def __call__(self) -> dt.date:
        return self.day


def mismatch_reply(*items: tuple[str, str, str]) -> CaptionReply:
    return CaptionReply.model_validate(
        {"mismatches": [{"field": f, "claimed": c, "original": o} for f, c, o in items]}
    )
