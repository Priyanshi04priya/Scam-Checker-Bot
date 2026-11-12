"""Fixed user-facing strings, loaded from one YAML file per language and validated at startup."""

from pathlib import Path
from typing import Annotated, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

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


class VerdictTexts(_Strict):
    bands: BandTexts
    labels: LabelTexts
    stub_note: Text


class LocaleTexts(_Strict):
    """Every string one language must provide; a missing or unknown key fails validation."""

    commands: CommandTexts
    checking: Text
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
