"""Record and replay vendor traffic so a costly run can be scored again for free.

``ReplayStore`` is a JSON file of responses. ``RecordingLLM`` and ``RecordingCaller`` wrap the
real clients: a stored response is returned without calling the vendor, a miss calls it and
stores the answer (unless ``replay_only``, which refuses). ``DisablingCaller`` and
``DisabledLLMClient`` simulate a vendor outage.

Run ``python -m eval.replay build-seed`` to regenerate the small file the offline mode uses.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from scambot.agent.executor import ToolCaller, ToolSession
from scambot.core.contracts import CheckerResult
from scambot.core.llm import DisabledLLMClient, LLMClient, LLMError, LLMImage

SEED_REPLAY: Final = Path(__file__).parent / "data" / "seed" / "replay_offline.json"
SEED_TODAY: Final = "2026-01-15"
IGNORED_ARGUMENTS: Final = ("trace_id", "media_key")

# Which checker tools a vendor outage switches off.
VENDOR_TOOLS: Final[dict[str, tuple[str, ...]]] = {
    "google_vision": ("check_provenance",),
    "sightengine": ("detect_deepfake",),
    "google_factcheck": ("fact_check_claim",),
    "google_safe_browsing": ("analyze_scam_patterns",),
    "voice": ("detect_voice_clone",),
    "anthropic": (),
}
VENDORS: Final = tuple(VENDOR_TOOLS)


_SECRET_PARAM_RE: Final = re.compile(
    r"(?i)([?&](?:key|api_key|apikey|api-key|token|access_token|auth|secret)=)[^&\s\"']+"
)
_BEARER_RE: Final = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}")


def scrub(value: Any) -> Any:
    """Remove key-like query parameters and bearer tokens from a value before it is stored."""
    text = json.dumps(value, ensure_ascii=False)
    cleaned = _BEARER_RE.sub(r"\1 REDACTED", _SECRET_PARAM_RE.sub(r"\1REDACTED", text))
    return value if cleaned == text else json.loads(cleaned)


class ReplayMissError(LLMError):
    """A replay-only run asked for a response that was never recorded."""


def llm_key(
    prompt_name: str, variables: Mapping[str, str], images: Sequence[LLMImage] | None = None
) -> str:
    digests = [hashlib.sha256(image.data).hexdigest() for image in images or ()]
    payload = json.dumps(
        [prompt_name, sorted(variables.items()), digests], ensure_ascii=False, separators=(",", ":")
    )
    return "llm:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tool_key(name: str, arguments: Mapping[str, Any]) -> str:
    """Tool name plus a hash of the arguments, without the trace ID and the media key."""
    kept = {k: v for k, v in arguments.items() if k not in IGNORED_ARGUMENTS}
    encoded = json.dumps(
        kept, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return f"tool:{name}:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ReplayStore:
    """Recorded responses by key, saved atomically after every new entry."""

    def __init__(self, path: Path | None = None, *, autosave: bool = True) -> None:
        self.path = path
        self.autosave = autosave
        self.entries: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        self.recorded = 0
        if path is not None and path.is_file():
            self.entries = json.loads(path.read_text(encoding="utf-8")).get("entries", {})

    def get(self, key: str) -> Any | None:
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def put(self, key: str, value: Any) -> None:
        self.entries[key] = scrub(value)
        self.recorded += 1
        if self.autosave:
            self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {"version": 1, "entries": self.entries},
            ensure_ascii=False,
            indent=0,
            sort_keys=True,
        )
        body += "\n"
        handle, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(body)
            Path(tmp).chmod(0o644)
            Path(tmp).replace(self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class RecordingLLM:
    """Answers from the store; on a miss calls ``inner`` (if any, and allowed) and records."""

    def __init__(
        self, inner: LLMClient | None, store: ReplayStore, *, replay_only: bool = False
    ) -> None:
        self._inner = inner
        self._store = store
        self._replay_only = replay_only

    async def structured[T: BaseModel](
        self,
        prompt_name: str,
        variables: Mapping[str, str],
        schema: type[T],
        images: Sequence[LLMImage] | None = None,
    ) -> T:
        key = llm_key(prompt_name, variables, images)
        stored = self._store.get(key)
        if stored is not None:
            return schema.model_validate(stored["response"])
        if self._inner is None or self._replay_only:
            msg = f"no recorded response for {prompt_name}"
            raise ReplayMissError(msg)
        result = await self._inner.structured(prompt_name, variables, schema, images)
        self._store.put(key, {"prompt": prompt_name, "response": result.model_dump(mode="json")})
        return result


class ReplayLLM(RecordingLLM):
    """Replay only: a miss is an ``LLMError``, like an LLM outage."""

    def __init__(self, store: ReplayStore) -> None:
        super().__init__(None, store, replay_only=True)


class _RecordingSession:
    def __init__(self, inner: ToolSession | None, store: ReplayStore, replay_only: bool) -> None:
        self._inner = inner
        self._store = store
        self._replay_only = replay_only

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        key = tool_key(name, arguments)
        stored = self._store.get(key)
        if stored is not None:
            result: dict[str, Any] = stored["result"]
            return result
        if self._inner is None or self._replay_only:
            msg = f"no recorded response for {name}"
            raise ReplayMissError(msg)
        raw = await self._inner.call(name, arguments)
        clean = CheckerResult.model_validate(raw).model_dump(mode="json")
        self._store.put(key, {"tool": name, "result": clean})
        return raw


class RecordingCaller:
    """A tool caller that replays recorded checker results and records new ones."""

    def __init__(
        self, inner: ToolCaller | None, store: ReplayStore, *, replay_only: bool = False
    ) -> None:
        self._inner = inner
        self._store = store
        self._replay_only = replay_only

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ToolSession]:
        if self._inner is None or self._replay_only:
            yield _RecordingSession(None, self._store, True)
            return
        async with self._inner.session() as session:
            yield _RecordingSession(session, self._store, False)


class _DisablingSession:
    def __init__(self, inner: ToolSession, disabled: Collection[str]) -> None:
        self._inner = inner
        self._disabled = disabled

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in self._disabled:
            result = CheckerResult(
                checker=name,
                status="error",
                signal="none",
                confidence=0.0,
                duration_ms=0,
                error="vendor_disabled",
            )
            return result.model_dump(mode="json")
        return await self._inner.call(name, arguments)


class DisablingCaller:
    """Returns an error result for the tools whose vendor is switched off."""

    def __init__(self, inner: ToolCaller, vendors: Collection[str]) -> None:
        self._inner = inner
        self.tools = frozenset(t for v in vendors for t in VENDOR_TOOLS.get(v, ()))

    def session(self) -> AbstractAsyncContextManager[ToolSession]:
        return self._wrap()

    @asynccontextmanager
    async def _wrap(self) -> AsyncIterator[ToolSession]:
        async with self._inner.session() as session:
            yield _DisablingSession(session, self.tools)


def disabled_llm() -> DisabledLLMClient:
    return DisabledLLMClient()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.replay")
    parser.add_argument("command", choices=["build-seed"])
    parser.add_argument("--data", type=Path, default=SEED_REPLAY.parent)
    parser.add_argument("--out", type=Path, default=SEED_REPLAY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from eval.backends import build_seed_replay  # noqa: PLC0415 - heavy imports only here

    count = asyncio.run(build_seed_replay(args.data, args.out))
    sys.stdout.write(f"wrote {count} entries to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
