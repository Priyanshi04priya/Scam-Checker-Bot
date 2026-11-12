"""Fixed user-facing strings, loaded from one YAML file per language and validated at startup."""

from pathlib import Path
from typing import Annotated, Final, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scambot.core.settings import CONFIG_DIR

DEFAULT_LANGUAGE: Final = "en"
LOCALES_DIR: Final = CONFIG_DIR / "locales"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Text = Annotated[str, Field(min_length=1)]


class CommandTexts(_Strict):
    start: Text
    help: Text
    privacy: Text
    unknown: Text


class BandTexts(_Strict):
    high: Text
    medium: Text
    low: Text
    unknown: Text


class LabelTexts(_Strict):
    false_misleading: Text
    likely_scam: Text
    old_or_out_of_context: Text
    possibly_ai: Text
    no_issues: Text
    could_not_verify: Text


class CheckerTexts(_Strict):
    """Everyday names of the checks, used when a check could not run."""

    analyze_scam_patterns: Text
    fact_check_claim: Text
    check_provenance: Text
    check_caption_context: Text
    detect_deepfake: Text
    detect_voice_clone: Text


class SignalTexts(_Strict):
    """Everyday descriptions of each scam warning sign."""

    kyc_update: Text
    lottery_prize: Text
    job_offer: Text
    digital_arrest: Text
    electricity_disconnection: Text
    courier_customs: Text
    investment_doubling: Text
    official_impersonation: Text
    urgency: Text
    threat: Text
    payment_request: Text
    otp_request: Text
    phone_number: Text
    upi_id: Text
    shortened_url: Text
    lookalike_domain: Text
    unsafe_url: Text


SIGNALS_PLACEHOLDER: Final = "{signals}"
CHECKS_PLACEHOLDER: Final = "{checks}"


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, BaseModel):
        return [text for item in value.model_dump().values() for text in _strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    return []


class VerdictTexts(_Strict):
    """Placeholders are replaced with ``str.replace``; no other braces are allowed."""

    bands: BandTexts
    labels: LabelTexts
    explanations: LabelTexts
    signals: SignalTexts
    signals_generic: Text
    checkers: CheckerTexts
    unavailable: Text
    sources: Text

    @model_validator(mode="after")
    def _check_placeholders(self) -> Self:
        if SIGNALS_PLACEHOLDER not in self.explanations.likely_scam:
            msg = f"explanations.likely_scam must contain {SIGNALS_PLACEHOLDER}"
            raise ValueError(msg)
        if CHECKS_PLACEHOLDER not in self.unavailable:
            msg = f"unavailable must contain {CHECKS_PLACEHOLDER}"
            raise ValueError(msg)
        rest = self.model_copy(
            update={
                "explanations": self.explanations.model_copy(
                    update={
                        "likely_scam": self.explanations.likely_scam.replace(
                            SIGNALS_PLACEHOLDER, ""
                        )
                    }
                ),
                "unavailable": self.unavailable.replace(CHECKS_PLACEHOLDER, ""),
            }
        )
        if any("{" in text or "}" in text for text in _strings(rest)):
            msg = "verdict texts may not contain braces other than the known placeholders"
            raise ValueError(msg)
        return self


class LocaleTexts(_Strict):
    """Every string one language must provide; a missing or unknown key fails validation."""

    commands: CommandTexts
    checking: Text
    timeout: Text
    verdict: VerdictTexts
    disclaimer: Text


class Locales:
    """All loaded languages, with English as the fallback."""

    def __init__(self, texts: dict[str, LocaleTexts]) -> None:
        if DEFAULT_LANGUAGE not in texts:
            msg = f"the {DEFAULT_LANGUAGE!r} locale is required"
            raise ValueError(msg)
        self._texts = dict(texts)

    @property
    def languages(self) -> frozenset[str]:
        return frozenset(self._texts)

    def resolve(self, language: str | None) -> str:
        """Return the loaded language that best matches ``language`` (``hi-IN`` → ``hi``)."""
        if language:
            primary = language.replace("_", "-").split("-", 1)[0].lower()
            if primary in self._texts:
                return primary
        return DEFAULT_LANGUAGE

    def get(self, language: str | None) -> LocaleTexts:
        return self._texts[self.resolve(language)]


def load_locales(directory: Path = LOCALES_DIR) -> Locales:
    """Load and validate every ``*.yaml`` file in ``directory``; the file stem is the language."""
    texts: dict[str, LocaleTexts] = {}
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            texts[path.stem] = LocaleTexts.model_validate(data)
        except ValueError as exc:
            msg = f"invalid locale file {path.name}: {exc}"
            raise ValueError(msg) from exc
    return Locales(texts)
