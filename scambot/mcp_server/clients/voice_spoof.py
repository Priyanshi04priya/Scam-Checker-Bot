"""Synthetic-voice scoring backends.

A detector rates a short 16 kHz mono clip with the probability that the voice is synthetic.
Scores are only ever used as an uncertain signal.
"""

import importlib
import importlib.util
import threading
from collections.abc import Callable
from typing import Any, Final, Protocol

import httpx

from scambot.core.config import VoiceThresholds
from scambot.core.logging import get_logger
from scambot.core.settings import Settings
from scambot.preprocess.stt import SAMPLE_RATE, Audio

_log = get_logger(component="voice_spoof")

Loader = Callable[[], tuple[Any, Any]]
_REQUIRED_PACKAGES: Final = ("torch", "transformers")


class VoiceSpoofError(Exception):
    """A detector failed; the message is a short fixed code."""


class VoiceSpoofDetector(Protocol):
    name: str

    def score(self, audio: Audio) -> float:
        """Probability in [0, 1] that the clip is synthetic (blocking)."""
        ...

    def warm_up(self) -> None:
        """Load the model now; raises ``VoiceSpoofError`` when that is not possible."""
        ...


class HfAudioSpoofDetector:
    """A Hugging Face audio-classification model run on the CPU; it loads on first use."""

    def __init__(  # noqa: PLR0913
        self,
        model_id: str,
        *,
        revision: str | None,
        spoof_labels: list[str],
        window_s: float,
        threads: int = 0,
        loader: Loader | None = None,
    ) -> None:
        self.name = f"huggingface:{model_id}"
        self._model_id = model_id
        self._revision = revision
        self._spoof_labels = {label.lower() for label in spoof_labels}
        self._max_samples = round(window_s * SAMPLE_RATE)
        self._threads = threads
        self._loader = loader or self._load_hf
        self._loaded: tuple[Any, Any, list[int]] | None = None
        self._lock = threading.Lock()

    def _load_hf(self) -> tuple[Any, Any]:
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError:
            raise VoiceSpoofError("not_configured") from None
        try:
            if self._threads:
                torch.set_num_threads(self._threads)
            extractor = transformers.AutoFeatureExtractor.from_pretrained(
                self._model_id, revision=self._revision, trust_remote_code=False
            )
            model = transformers.AutoModelForAudioClassification.from_pretrained(
                self._model_id,
                revision=self._revision,
                use_safetensors=True,
                trust_remote_code=False,
            )
        except Exception:
            raise VoiceSpoofError("load_failed") from None
        return extractor, model

    def _ensure(self) -> tuple[Any, Any, list[int]]:
        with self._lock:
            if self._loaded is None:
                extractor, model = self._loader()
                try:
                    model.eval()
                    id2label = model.config.id2label
                    spoof_ids = [
                        int(index)
                        for index, label in id2label.items()
                        if str(label).lower() in self._spoof_labels
                    ]
                except Exception:
                    raise VoiceSpoofError("load_failed") from None
                if not spoof_ids:
                    raise VoiceSpoofError("unknown_labels")
                self._loaded = (extractor, model, spoof_ids)
            return self._loaded

    def warm_up(self) -> None:
        self._ensure()

    def score(self, audio: Audio) -> float:
        extractor, model, spoof_ids = self._ensure()
        try:
            torch = importlib.import_module("torch")
            inputs = extractor(
                audio[: self._max_samples], sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            with torch.inference_mode():
                logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            return float(sum(float(probabilities[i]) for i in spoof_ids))
        except Exception:
            raise VoiceSpoofError("inference_failed") from None


class RealityDefenderDetector:
    """A hosted detector. Not usable yet.

    The request format is unverified because the vendor documentation could not be reached.
    Enabling a hosted detector sends the audio to a third party, so the privacy notice must
    name that service before this is switched on.
    """

    name = "reality_defender"

    def __init__(self, http: httpx.Client, api_key: str) -> None:
        if not api_key.strip():
            msg = "an API key is required"
            raise ValueError(msg)
        self._http = http
        self._api_key = api_key

    def score(self, audio: Audio) -> float:
        raise VoiceSpoofError("not_implemented")

    def warm_up(self) -> None:
        return None


def create_voice_detector(settings: Settings, cfg: VoiceThresholds) -> VoiceSpoofDetector | None:
    """The configured detector, or None (with a logged reason) when it cannot be used."""
    backend = settings.voice_backend
    if backend == "none":
        _log.info("voice_detector_unavailable", reason="disabled")
        return None
    if backend == "reality_defender":
        key = settings.reality_defender_api_key.get_secret_value()
        if not key.strip():
            _log.warning("voice_detector_unavailable", reason="no_api_key")
            return None
        # The hosted detector cannot score yet, so it is not offered.
        _log.warning("voice_detector_unavailable", reason="hosted_not_implemented")
        return None
    if any(importlib.util.find_spec(name) is None for name in _REQUIRED_PACKAGES):
        _log.warning("voice_detector_unavailable", reason="no_package")
        return None
    return HfAudioSpoofDetector(
        settings.voice_model_id,
        revision=settings.voice_model_revision or None,
        spoof_labels=cfg.spoof_labels,
        window_s=cfg.window_s,
        threads=settings.voice_cpu_threads,
    )
