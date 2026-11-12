"""Tests for the sentence-embedding wrapper (a stub model; torch is never needed)."""

import importlib.util

import pytest

from scambot.core.settings import Settings
from scambot.factcheck_index import embed
from scambot.factcheck_index.embed import (
    SentenceTransformerEmbedder,
    create_embedder,
    sentence_transformers_available,
)
from scambot.factcheck_index.records import passage_text
from tests.fakes import StubEncoder
from tests.samples import LAPTOP_FACTCHECK


def stub_embedder(encoder: StubEncoder, loads: list[str]) -> SentenceTransformerEmbedder:
    def loader(name: str) -> StubEncoder:
        loads.append(name)
        return encoder

    return SentenceTransformerEmbedder("e5", batch_size=4, loader=loader)


async def test_prefixes_and_encode_arguments() -> None:
    encoder, loads = StubEncoder(), list[str]()
    embedder = stub_embedder(encoder, loads)

    queries = await embedder.embed_queries(["claim one", "claim two"])
    passages = await embedder.embed_passages(["article"])

    assert len(queries) == 2
    assert all(len(row) == 384 for row in queries + passages)
    assert [inputs for inputs, _ in encoder.calls] == [
        ["query: claim one", "query: claim two"],
        ["passage: article"],
    ]
    assert encoder.calls[0][1] == {
        "batch_size": 4,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    assert loads == ["e5"]  # loaded once


async def test_empty_input_does_not_load_the_model() -> None:
    loads: list[str] = []
    embedder = stub_embedder(StubEncoder(), loads)

    assert await embedder.embed_queries([]) == []
    assert loads == []


async def test_wrong_dimension_is_rejected() -> None:
    embedder = stub_embedder(StubEncoder(dim=12), [])

    with pytest.raises(ValueError, match="384"):
        await embedder.embed_passages(["x"])


def test_create_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert sentence_transformers_available() is False
    assert create_embedder(Settings(_env_file=None)) is None

    monkeypatch.setattr(embed, "sentence_transformers_available", lambda: True)
    created = create_embedder(Settings(_env_file=None))
    assert isinstance(created, SentenceTransformerEmbedder)
    assert created._model is None  # not loaded until used


def test_default_loader_needs_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> object:
        raise ModuleNotFoundError(embed.PACKAGE)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(ModuleNotFoundError):
        SentenceTransformerEmbedder("e5").load()


def test_passage_text() -> None:
    assert passage_text(LAPTOP_FACTCHECK) == f"{LAPTOP_FACTCHECK.claim}. {LAPTOP_FACTCHECK.title}"
    same = LAPTOP_FACTCHECK.model_copy(update={"title": LAPTOP_FACTCHECK.claim})
    assert passage_text(same) == LAPTOP_FACTCHECK.claim
