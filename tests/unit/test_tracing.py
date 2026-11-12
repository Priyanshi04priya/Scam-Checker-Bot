"""Tests for the trace ID context variable."""

import asyncio

import pytest

from scambot.core import tracing


def test_no_trace_id_by_default() -> None:
    assert tracing.get_trace_id() is None


def test_new_trace_id_is_unique_hex() -> None:
    first, second = tracing.new_trace_id(), tracing.new_trace_id()

    assert first != second
    assert len(first) == 32
    int(first, 16)


def test_set_and_reset_trace_id() -> None:
    token = tracing.set_trace_id("abc")
    assert tracing.get_trace_id() == "abc"

    tracing.reset_trace_id(token)
    assert tracing.get_trace_id() is None


def test_set_trace_id_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tracing.set_trace_id("")


def test_trace_context_generates_and_restores() -> None:
    with tracing.trace_context() as outer:
        assert tracing.get_trace_id() == outer
        with tracing.trace_context("inner") as inner:
            assert inner == "inner"
            assert tracing.get_trace_id() == "inner"
        assert tracing.get_trace_id() == outer
    assert tracing.get_trace_id() is None


def test_trace_context_replaces_an_empty_id() -> None:
    with tracing.trace_context("") as current:
        assert len(current) == 32
        assert tracing.get_trace_id() == current
    assert tracing.get_trace_id() is None


def test_trace_context_restores_after_exception() -> None:
    with pytest.raises(RuntimeError), tracing.trace_context("t-1"):
        raise RuntimeError
    assert tracing.get_trace_id() is None


async def test_concurrent_tasks_keep_their_own_trace_id() -> None:
    async def worker(trace_id: str) -> str | None:
        with tracing.trace_context(trace_id):
            await asyncio.sleep(0)
            return tracing.get_trace_id()

    results = await asyncio.gather(worker("a"), worker("b"), worker("c"))

    assert list(results) == ["a", "b", "c"]


async def test_child_tasks_and_threads_inherit_the_trace_id() -> None:
    async def child() -> str | None:
        inherited = tracing.get_trace_id()
        with tracing.trace_context("child-only"):
            await asyncio.sleep(0)
        return inherited

    with tracing.trace_context("parent"):
        assert await asyncio.create_task(child()) == "parent"
        assert await asyncio.to_thread(tracing.get_trace_id) == "parent"
        assert tracing.get_trace_id() == "parent"


def test_add_trace_id_processor() -> None:
    assert tracing.add_trace_id(None, "info", {"event": "x"}) == {"event": "x"}

    with tracing.trace_context("t-2"):
        assert tracing.add_trace_id(None, "info", {"event": "x"}) == {
            "event": "x",
            "trace_id": "t-2",
        }
        assert tracing.add_trace_id(None, "info", {"event": "x", "trace_id": "own"}) == {
            "event": "x",
            "trace_id": "own",
        }
