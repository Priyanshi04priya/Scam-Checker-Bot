"""Tests for the synthetic-voice detector backends."""

from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from scambot.core.config import AppConfig
from scambot.core.settings import Settings
from scambot.mcp_server.clients import voice_spoof
from scambot.mcp_server.clients.voice_spoof import (
    HfAudioSpoofDetector,
    RealityDefenderDetector,
    VoiceSpoofError,
    create_voice_detector,
)


class StubModel:
    def __init__(self, labels: dict[int, str]) -> None:
        self.config = SimpleNamespace(id2label=labels)
        self.evaluated = False

    def eval(self) -> None:
        self.evaluated = True


def detector(labels: dict[int, str], **kwargs: Any) -> tuple[HfAudioSpoofDetector, StubModel]:
    model = StubModel(labels)
    built = HfAudioSpoofDetector(
        "org/model",
        revision=None,
        spoof_labels=["Fake", "spoof"],
        window_s=10,
        loader=lambda: (object(), model),
        **kwargs,
    )
    return built, model


def test_spoof_labels_are_matched_ignoring_case() -> None:
    built, model = detector({0: "real", 1: "FAKE"})

    built.warm_up()

    assert model.evaluated
    assert built.name == "huggingface:org/model"


def test_a_label_set_without_a_spoof_label_is_rejected() -> None:
    built, _ = detector({0: "bonafide", 1: "class_b"})

    with pytest.raises(VoiceSpoofError, match="unknown_labels"):
        built.warm_up()


def test_a_broken_loader_is_a_load_failure() -> None:
    def broken() -> tuple[Any, Any]:
        raise OSError

    built = HfAudioSpoofDetector(
        "org/model", revision=None, spoof_labels=["fake"], window_s=10, loader=broken
    )

    with pytest.raises(OSError):  # noqa: PT011 - the injected loader's own error passes through
        built.warm_up()


def test_missing_libraries_mean_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_import(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(voice_spoof, "importlib", SimpleNamespace(import_module=no_import))
    built = HfAudioSpoofDetector("org/model", revision=None, spoof_labels=["fake"], window_s=10)

    with pytest.raises(VoiceSpoofError, match="not_configured"):
        built.warm_up()


def test_scoring_failure_is_reported() -> None:
    built, _ = detector({0: "real", 1: "fake"})

    with pytest.raises(VoiceSpoofError, match="inference_failed"):
        built.score(np.zeros(16000, dtype=np.float32))


def test_reality_defender_is_a_stub() -> None:
    with pytest.raises(ValueError, match="key"):
        RealityDefenderDetector(httpx.Client(), " ")
    stub = RealityDefenderDetector(httpx.Client(), "key")

    stub.warm_up()
    with pytest.raises(VoiceSpoofError, match="not_implemented"):
        stub.score(np.zeros(10, dtype=np.float32))


def test_factory_returns_none_when_unavailable(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = app_config.thresholds.voice

    assert create_voice_detector(Settings(voice_backend="none"), cfg) is None
    assert create_voice_detector(Settings(voice_backend="reality_defender"), cfg) is None
    monkeypatch.setattr(
        voice_spoof, "importlib", SimpleNamespace(util=SimpleNamespace(find_spec=lambda _n: None))
    )
    assert create_voice_detector(Settings(voice_backend="huggingface"), cfg) is None


def test_factory_builds_the_configured_backends(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = app_config.thresholds.voice
    monkeypatch.setattr(
        voice_spoof,
        "importlib",
        SimpleNamespace(util=SimpleNamespace(find_spec=lambda _n: object())),
    )

    hf = create_voice_detector(Settings(voice_model_id="org/m", voice_model_revision=""), cfg)
    hosted = create_voice_detector(
        Settings(voice_backend="reality_defender", reality_defender_api_key=SecretStr("k")), cfg
    )

    assert hf is not None
    assert hf.name == "huggingface:org/m"
    assert hosted is None


def test_hf_load_pins_revision_and_forbids_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class _Auto:
        @staticmethod
        def from_pretrained(_model: str, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    fake = SimpleNamespace(
        AutoFeatureExtractor=_Auto, AutoModelForAudioClassification=_Auto, set_num_threads=print
    )
    monkeypatch.setattr(voice_spoof, "importlib", SimpleNamespace(import_module=lambda _name: fake))
    detector = voice_spoof.HfAudioSpoofDetector(
        "org/m", revision="abc", spoof_labels=["fake"], window_s=10
    )
    detector._load_hf()

    assert len(calls) == 2
    assert all(c["revision"] == "abc" and c["trust_remote_code"] is False for c in calls)
    assert calls[1]["use_safetensors"] is True
