"""Text embeddings for the fact-check index.

The sentence-transformers package is optional: without it the index is not searched, but
everything still imports and runs.
"""

import asyncio
import importlib
import importlib.util
import threading
from collections.abc import Callable, Sequence
from typing import Any, Final, Protocol

from scambot.core.logging import get_logger
from scambot.core.models import EMBEDDING_DIM
from scambot.core.settings import Settings

QUERY_PREFIX: Final = "query: "
PASSAGE_PREFIX: Final = "passage: "
PACKAGE: Final = "sentence_transformers"

_log = get_logger(component="embedder")


class Embedder(Protocol):
    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


class EncoderModel(Protocol):
    def encode(
        self,
        inputs: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any: ...


def _default_loader(model_name: str) -> EncoderModel:
    module = importlib.import_module(PACKAGE)
    model: EncoderModel = module.SentenceTransformer(model_name, device="cpu")
    return model


class SentenceTransformerEmbedder:
    """Unit-length E5 embeddings computed on the CPU in a worker thread."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        loader: Callable[[str], EncoderModel] | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._loader = loader or _default_loader
        self._model: EncoderModel | None = None
        self._lock = threading.Lock()

    def load(self) -> EncoderModel:
        """Load the model once (downloads it on first use)."""
        with self._lock:
            if self._model is None:
                self._model = self._loader(self._model_name)
            return self._model

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(QUERY_PREFIX, texts)

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(PASSAGE_PREFIX, texts)

    async def _embed(self, prefix: str, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, [prefix + text for text in texts])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        rows: list[list[float]] = (
            self.load()
            .encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            .tolist()
        )
        if len(rows) != len(texts) or any(len(row) != EMBEDDING_DIM for row in rows):
            msg = f"embeddings must have {EMBEDDING_DIM} dimensions"
            raise ValueError(msg)
        return [[float(value) for value in row] for row in rows]


def sentence_transformers_available() -> bool:
    return importlib.util.find_spec(PACKAGE) is not None


def create_embedder(settings: Settings) -> SentenceTransformerEmbedder | None:
    """The embedder (not yet loaded), or None when the optional package is missing."""
    if not sentence_transformers_available():
        _log.info("embedder_unavailable", reason="not_installed")
        return None
    return SentenceTransformerEmbedder(settings.embedding_model)
