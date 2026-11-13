"""Pipeline stacks for evaluation: an offline one made of fakes and a live one.

The offline stack runs the real rule checks, planner, verdict rules, reply composer, locales and
configuration, with test doubles for every vendor and a canned LLM read from a replay file. The
live stack reuses the production wiring. ``tests`` is imported only inside the functions that
need its fakes, so importing this module never needs the test folder.
"""

import json
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from arq.connections import ArqRedis
from pydantic import BaseModel, SecretStr

from eval.replay import (
    SEED_TODAY,
    VENDOR_TOOLS,
    DisablingCaller,
    RecordingCaller,
    RecordingLLM,
    ReplayLLM,
    ReplayStore,
    disabled_llm,
)
from eval.schema import load_dir
from scambot.agent.analyze import Analysis
from scambot.agent.executor import Executor, McpToolCaller, ToolCaller
from scambot.channels.base import ChannelAdapter
from scambot.core.cache import CheckerCache
from scambot.core.config import AppConfig, load_config
from scambot.core.llm import LLMClient, LLMImage, create_llm_client, load_prompts
from scambot.core.locales import load_locales
from scambot.core.media_store import LocalMediaStore
from scambot.core.settings import Settings
from scambot.factcheck_index.records import ArticleRecord
from scambot.mcp_server.checkers.factcheck import CandidateJudgment, RelevanceReply
from scambot.mcp_server.checkers.scam import ScamClassification
from scambot.mcp_server.server import CheckerDeps, build_server, create_deps
from scambot.preprocess.entities import NoopExpander
from scambot.preprocess.language import ScriptLanguageDetector
from scambot.preprocess.media import MediaEngines, create_engines
from scambot.worker.pipeline import PipelineDeps

if TYPE_CHECKING:
    from tests.fakes import FakeAdapter, InMemoryStore

Mode = Literal["offline", "live"]
_DEVANAGARI: Final = re.compile(r"[ऀ-ॿ]")
_WORD: Final = re.compile(r"[^\s.,;:!?()\"'।]+")
LLM_VENDOR: Final = "anthropic"
# Settings fields blanked to simulate a missing key, by vendor.
_VENDOR_SETTINGS: Final[dict[str, dict[str, Any]]] = {
    "anthropic": {"anthropic_api_key": SecretStr("")},
    "google_vision": {"google_api_key": SecretStr("")},
    "google_factcheck": {"google_api_key": SecretStr("")},
    "google_safe_browsing": {"google_api_key": SecretStr("")},
    "sightengine": {"sightengine_user": "", "sightengine_secret": SecretStr("")},
    "voice": {"voice_backend": "none"},
}


@dataclass
class Stack:
    """Everything one evaluation run needs, whatever the mode."""

    mode: Mode
    deps: PipelineDeps
    adapter: "FakeAdapter"
    store: "InMemoryStore"
    media_root: Path
    replay: ReplayStore
    disabled: frozenset[str] = frozenset()
    has_factcheck_index: bool = True
    cleanups: list[Callable[[], None]] = field(default_factory=list)
    closeables: list[Any] = field(default_factory=list)

    async def aclose(self) -> None:
        try:
            for item in self.closeables:
                await item.aclose()
        finally:
            for cleanup in self.cleanups:
                cleanup()


class SeedStubLLM:
    """The deterministic LLM that ``build-seed`` records into the offline replay file."""

    async def structured[T: BaseModel](
        self,
        prompt_name: str,
        variables: Mapping[str, str],
        schema: type[T],
        images: Sequence[LLMImage] | None = None,
    ) -> T:
        from scambot.core.llm import LLMError  # noqa: PLC0415

        if prompt_name == "analyze":
            content = variables["content"]
            language = "hi" if _DEVANAGARI.search(content) else "en"
            reply: BaseModel = Analysis(
                claims=[content.strip()[:300]],
                reply_language=language,
                claimed_date=None,
                claimed_place=None,
                extra_checkers=[],
                reason="stub",
            )
        elif prompt_name == "scam_classify":
            reply = ScamClassification(template="none", spans=[], confidence=0.0)
        elif prompt_name == "factcheck_relevance":
            reply = _stub_relevance(variables)
        else:
            msg = f"no stub for {prompt_name}"
            raise LLMError(msg)
        return schema.model_validate(reply.model_dump())


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.casefold()) if len(w) > 2}


def _stub_relevance(variables: Mapping[str, str]) -> RelevanceReply:
    claim = _words(variables["claim"])
    judgments: list[CandidateJudgment] = []
    for item in json.loads(variables["candidates_json"]):
        other = _words(item["claim_checked"])
        overlap = len(claim & other) / max(len(other), 1)
        relation = "same" if overlap >= 0.8 else "different"
        judgments.append(
            CandidateJudgment.model_validate(
                {"candidate": item["candidate"], "relation": relation, "polarity": "false"}
            )
        )
    return RelevanceReply(judgments=judgments)


def load_articles(path: Path) -> list[ArticleRecord]:
    records: list[ArticleRecord] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(ArticleRecord.model_validate_json(line))
    return records


def _fake_engines(config: AppConfig) -> tuple[MediaEngines, ThreadPoolExecutor]:
    from tests.fakes import FakeAudio, FakeKeyframes, FakeOcr, FakeStt  # noqa: PLC0415

    pool = ThreadPoolExecutor(max_workers=2)
    return (
        MediaEngines(
            ocr=FakeOcr(),
            stt=FakeStt(),
            audio=FakeAudio(),
            keyframes=FakeKeyframes(),
            language=ScriptLanguageDetector(),
            pool=pool,
            limits=config.thresholds.media,
        ),
        pool,
    )


async def _fake_cache(config: AppConfig) -> CheckerCache:
    from fakeredis import FakeAsyncRedis, FakeServer  # noqa: PLC0415

    redis = ArqRedis(connection_pool=FakeAsyncRedis(server=FakeServer()).connection_pool)
    cache = config.thresholds.cache
    return CheckerCache(redis, cache.checker_ttl_s, cache.inconclusive_ttl_s)


async def build_offline_stack(  # noqa: PLR0913
    *,
    replay: ReplayStore,
    disabled: Sequence[str] = (),
    config: AppConfig | None = None,
    articles: Sequence[ArticleRecord] = (),
    llm: LLMClient | None = None,
    wrap_caller: Callable[[ToolCaller], ToolCaller] | None = None,
    work_dir: Path | None = None,
) -> Stack:
    """The key-less stack: fakes for every vendor, canned LLM answers from ``replay``."""
    from tests.fakes import (  # noqa: PLC0415
        FakeAdapter,
        FakeImageAnalyzer,
        FakePageFetcher,
        FakeVoiceDetector,
        FakeWebSearch,
        InMemoryStore,
        KeywordEmbedder,
        seed,
        speech_vad,
    )

    from scambot.factcheck_index.search import InMemoryFactcheckIndex  # noqa: PLC0415
    from scambot.mcp_server.checkers.caption import CaptionChecker  # noqa: PLC0415
    from scambot.mcp_server.checkers.deepfake import DeepfakeChecker  # noqa: PLC0415
    from scambot.mcp_server.checkers.factcheck import (  # noqa: PLC0415
        FactCheckChecker,
        FactCheckSources,
    )
    from scambot.mcp_server.checkers.provenance import ProvenanceChecker  # noqa: PLC0415
    from scambot.mcp_server.checkers.scam import ScamChecker  # noqa: PLC0415
    from scambot.mcp_server.checkers.voice_clone import VoiceCloneChecker  # noqa: PLC0415
    from scambot.mcp_server.media import MediaFiles  # noqa: PLC0415

    config = config or load_config()
    off = frozenset(disabled)
    root = work_dir or Path(tempfile.mkdtemp(prefix="scambot-eval-"))
    media_root = root / "media"
    media_store = LocalMediaStore(media_root)
    pipeline_llm: LLMClient = llm or ReplayLLM(replay)
    if LLM_VENDOR in off:
        pipeline_llm = disabled_llm()
    index = InMemoryFactcheckIndex()
    embedder = KeywordEmbedder()
    if articles:
        await seed(index, embedder, *articles)
    media = MediaFiles(media_store)
    deps = CheckerDeps(
        scam=ScamChecker(config.scam_patterns, config.thresholds.scam, pipeline_llm, None),
        factcheck=FactCheckChecker(
            FactCheckSources(api=None, index=index, embedder=embedder),
            pipeline_llm,
            config.thresholds.factcheck,
            config.sources,
        ),
        provenance=ProvenanceChecker(
            FakeWebSearch(),
            FakePageFetcher(),
            media,
            config.thresholds.provenance,
            config.sources,
        ),
        caption=CaptionChecker(
            pipeline_llm, config.thresholds.caption, config.thresholds.provenance
        ),
        deepfake=DeepfakeChecker(FakeImageAnalyzer(), media, config.thresholds.deepfake),
        voice=VoiceCloneChecker(FakeVoiceDetector(), media, config.thresholds.voice, speech_vad()),
    )
    caller: ToolCaller = McpToolCaller(
        build_server(deps), config.thresholds.pipeline.mcp_connect_timeout_s
    )
    if off & set(VENDOR_TOOLS) - {LLM_VENDOR}:
        caller = DisablingCaller(caller, off)
    if wrap_caller is not None:
        caller = wrap_caller(caller)
    engines, pool = _fake_engines(config)
    adapter = FakeAdapter()
    store = InMemoryStore()
    pipeline = PipelineDeps(
        adapters={"telegram": adapter},
        locales=load_locales(),
        config=config,
        llm=pipeline_llm,
        executor=Executor(caller, cache=await _fake_cache(config)),
        store=store,
        expander=NoopExpander(),
        media_store=media_store,
        media=engines,
    )
    stack = Stack("offline", pipeline, adapter, store, media_root, replay, off)
    stack.cleanups.append(lambda: pool.shutdown(wait=True))
    if work_dir is None:
        stack.cleanups.append(lambda: shutil.rmtree(root, ignore_errors=True))
    return stack


def missing_keys(settings: Settings, disabled: Sequence[str] = ()) -> list[str]:
    """Names of the keys a live run needs and does not have (a disabled vendor is not needed)."""
    off = set(disabled)
    needed = {
        "anthropic": ("ANTHROPIC_API_KEY", bool(settings.anthropic_api_key.get_secret_value())),
        "google_vision": ("GOOGLE_API_KEY", bool(settings.google_api_key.get_secret_value())),
        "sightengine": (
            "SIGHTENGINE_USER and SIGHTENGINE_SECRET",
            bool(settings.sightengine_user and settings.sightengine_secret.get_secret_value()),
        ),
    }
    return [name for vendor, (name, present) in needed.items() if vendor not in off and not present]


def disable_in_settings(settings: Settings, vendors: Sequence[str]) -> Settings:
    """A copy of ``settings`` with the vendors' keys blanked (the production missing-key path)."""
    update: dict[str, Any] = {}
    for vendor in vendors:
        update.update(_VENDOR_SETTINGS.get(vendor, {}))
    return settings.model_copy(update=update)


async def build_live_stack(  # noqa: PLR0913
    settings: Settings,
    config: AppConfig,
    *,
    replay: ReplayStore,
    replay_only: bool = False,
    disabled: Sequence[str] = (),
    database_url: str | None = None,
    work_dir: Path | None = None,
) -> Stack:
    """The production checkers (``create_deps``) behind a recording layer.

    Not exercised by the automated tests: it needs real keys. Status and adapter are fakes so
    that nothing is sent to a chat platform.
    """
    from tests.fakes import FakeAdapter, InMemoryStore  # noqa: PLC0415

    root = work_dir or Path(tempfile.mkdtemp(prefix="scambot-eval-"))
    live = disable_in_settings(settings, disabled).model_copy(
        update={"media_dir": root / "media"}
        | ({"database_url": SecretStr(database_url)} if database_url else {})
    )
    checker_deps, closeables = create_deps(live, config)
    llm, sdk = create_llm_client(live, load_prompts())
    if sdk is not None:
        closeables.append(sdk)
    inner = McpToolCaller(
        build_server(checker_deps), config.thresholds.pipeline.mcp_connect_timeout_s
    )
    off = frozenset(disabled)
    base: ToolCaller = DisablingCaller(inner, off) if off & set(VENDOR_TOOLS) else inner
    caller = RecordingCaller(base, replay, replay_only=replay_only)
    recording_llm = RecordingLLM(llm, replay, replay_only=replay_only)
    engines = create_engines(live, config.thresholds.media)
    adapter = FakeAdapter()
    store = InMemoryStore()
    adapters: dict[str, ChannelAdapter] = {"telegram": adapter}
    pipeline = PipelineDeps(
        adapters=adapters,
        locales=load_locales(),
        config=config,
        llm=recording_llm,
        executor=Executor(caller, cache=await _fake_cache(config)),
        store=store,
        expander=NoopExpander(),
        media_store=LocalMediaStore(root / "media"),
        media=engines,
    )
    stack = Stack(
        "live", pipeline, adapter, store, root / "media", replay, off, database_url is not None
    )
    stack.closeables.extend(closeables)
    stack.cleanups.append(lambda: engines.pool.shutdown(wait=True))
    return stack


async def build_seed_replay(data: Path, out: Path) -> int:
    """Run the offline stack once with the stub LLM and save every answer it gave."""
    from eval.tasks import run_item  # noqa: PLC0415 - tasks imports this module

    items = load_dir(data)
    store = ReplayStore(out, autosave=False)
    store.entries.clear()
    stack = await build_offline_stack(
        replay=store,
        llm=RecordingLLM(SeedStubLLM(), store),
        articles=load_articles(data / "synthetic_articles.jsonl"),
    )
    try:
        for item in sorted(items, key=lambda i: i.id):
            await run_item(stack, item, today=SEED_TODAY)
    finally:
        await stack.aclose()
    store.save()
    return len(store.entries)


def with_thresholds(config: AppConfig, **updates: Any) -> AppConfig:
    """A copy of ``config`` with top-level threshold sections replaced (memory only)."""
    return replace(config, thresholds=config.thresholds.model_copy(update=updates))
