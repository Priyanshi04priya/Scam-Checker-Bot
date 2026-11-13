"""Recording, replaying and simulating outages."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from eval.replay import (
    DisablingCaller,
    RecordingCaller,
    RecordingLLM,
    ReplayLLM,
    ReplayMissError,
    ReplayStore,
    disabled_llm,
    llm_key,
    tool_key,
)
from pydantic import BaseModel

from scambot.agent.executor import ToolSession
from scambot.core.contracts import CheckerResult
from scambot.core.llm import LLMError, LLMUnavailableError


class Answer(BaseModel):
    text: str


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def structured(
        self, prompt_name: str, variables: Any, schema: Any, images: Any = None
    ) -> Any:
        self.calls += 1
        return schema(text=f"{prompt_name}:{variables['x']}")


class CountingTools:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ToolSession]:
        outer = self

        class Session:
            async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                outer.calls.append(name)
                return CheckerResult(
                    checker=name, status="ok", signal="none", confidence=0.5, duration_ms=3
                ).model_dump(mode="json")

        yield Session()


async def test_record_then_replay_gives_identical_objects(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path / "r.json")
    inner = CountingLLM()
    first = await RecordingLLM(inner, store).structured("p", {"x": "1"}, Answer)
    again = await RecordingLLM(inner, ReplayStore(tmp_path / "r.json")).structured(
        "p", {"x": "1"}, Answer
    )
    assert first == again
    assert inner.calls == 1


async def test_replay_only_miss_raises(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path / "r.json")
    with pytest.raises(ReplayMissError):
        await RecordingLLM(CountingLLM(), store, replay_only=True).structured(
            "p", {"x": "1"}, Answer
        )
    with pytest.raises(LLMError):
        await ReplayLLM(store).structured("p", {"x": "1"}, Answer)
    assert store.misses == 2


async def test_tool_recording_and_miss(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path / "t.json")
    tools = CountingTools()
    args = {"text": "hi", "trace_id": "a"}
    async with RecordingCaller(tools, store).session() as session:
        first = await session.call("analyze_scam_patterns", args)
    async with RecordingCaller(tools, store).session() as session:
        second = await session.call("analyze_scam_patterns", {**args, "trace_id": "b"})
    assert first == second
    assert tools.calls == ["analyze_scam_patterns"]
    async with RecordingCaller(tools, store, replay_only=True).session() as session:
        with pytest.raises(ReplayMissError):
            await session.call("analyze_scam_patterns", {"text": "other"})
        with pytest.raises(ReplayMissError):
            await session.call("fact_check_claim", args)


def test_keys_ignore_trace_id_and_media_key_but_not_content() -> None:
    a = tool_key("t", {"text": "x", "trace_id": "1", "media_key": "m1"})
    b = tool_key("t", {"trace_id": "2", "text": "x", "media_key": "m2"})
    assert a == b
    assert a != tool_key("t", {"text": "y"})
    assert a != tool_key("u", {"text": "x"})
    assert llm_key("p", {"a": "1", "b": "2"}) == llm_key("p", {"b": "2", "a": "1"})
    assert llm_key("p", {"a": "1"}) != llm_key("q", {"a": "1"})


async def test_disabling_caller_and_llm() -> None:
    tools = CountingTools()
    caller = DisablingCaller(tools, ["sightengine", "google_factcheck"])
    async with caller.session() as session:
        dead = await session.call("detect_deepfake", {})
        dead_fc = await session.call("fact_check_claim", {})
        alive = await session.call("analyze_scam_patterns", {})
    assert dead["status"] == dead_fc["status"] == "error"
    assert dead["error"] == "vendor_disabled"
    assert alive["status"] == "ok"
    assert tools.calls == ["analyze_scam_patterns"]
    with pytest.raises(LLMUnavailableError):
        await disabled_llm().structured("p", {}, Answer)


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path / "deep" / "r.json")
    store.put("k", {"v": 1})
    store.put("k2", {"v": 2})
    assert json.loads((tmp_path / "deep" / "r.json").read_text())["entries"]["k2"] == {"v": 2}
    assert [p.name for p in (tmp_path / "deep").iterdir()] == ["r.json"]
    assert store.recorded == 2
    assert store.get("k") == {"v": 1}
    assert store.hits == 1


def test_save_failure_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ReplayStore(tmp_path / "r.json", autosave=False)
    store.entries["k"] = object()  # not JSON serialisable
    with pytest.raises(TypeError):
        store.save()
    assert list(tmp_path.iterdir()) == []


def test_stored_entries_never_keep_keys_or_bearer_tokens() -> None:
    from eval.replay import ReplayStore  # noqa: PLC0415

    store = ReplayStore(None)
    store.put(
        "k",
        {"url": "https://x.example/a?q=1&key=AIzaSECRET123&z=2", "h": "Bearer abcdef123456789"},
    )

    dumped = str(store.entries)
    assert "AIzaSECRET123" not in dumped
    assert "abcdef123456789" not in dumped
    assert "q=1" in dumped
