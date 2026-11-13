"""Which checkers to run for a request."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal, get_args

from scambot.agent.analyze import Analysis
from scambot.core.config import RoutingConfig
from scambot.core.contracts import CheckerName, JobKind, Preprocessed

SkipReason = Literal["disabled", "no_input", "dependency_not_planned"]

_VISUAL_KINDS: Final[frozenset[JobKind]] = frozenset({"image", "video"})
_AUDIO_KINDS: Final[frozenset[JobKind]] = frozenset({"audio", "voice", "video"})


@dataclass(frozen=True)
class PlannedChecker:
    name: CheckerName
    timeout_s: float
    depends_on: tuple[CheckerName, ...] = ()


@dataclass(frozen=True)
class Plan:
    kind: JobKind
    checkers: tuple[PlannedChecker, ...]
    skipped: Mapping[CheckerName, SkipReason] = field(default_factory=dict)

    @property
    def names(self) -> tuple[CheckerName, ...]:
        return tuple(checker.name for checker in self.checkers)


def _applicable(
    name: CheckerName, kind: JobKind, pre: Preprocessed, analysis: Analysis | None
) -> bool:
    if name == "analyze_scam_patterns":
        return bool(pre.text.strip())
    if name == "fact_check_claim":
        return analysis is not None and bool(analysis.claims)
    if name == "detect_voice_clone":
        return pre.has_media and kind in _AUDIO_KINDS
    if name == "check_caption_context":
        return pre.has_media and kind in _VISUAL_KINDS and bool(pre.text.strip())
    # Provenance and deepfake detection need an image or a video.
    return pre.has_media and kind in _VISUAL_KINDS


def build_plan(
    kind: JobKind, pre: Preprocessed, analysis: Analysis | None, routing: RoutingConfig
) -> Plan:
    """The route for ``kind`` plus the analysis' extras (which can add but never remove)."""
    candidates = list(routing.routes[kind])
    extras = set(analysis.extra_checkers) if analysis else set()
    candidates += [n for n in get_args(CheckerName) if n in extras and n not in candidates]
    planned: list[PlannedChecker] = []
    skipped: dict[CheckerName, SkipReason] = {}
    for name in candidates:
        config = routing.checkers[name]
        if not config.enabled:
            skipped[name] = "disabled"
        elif not _applicable(name, kind, pre, analysis):
            skipped[name] = "no_input"
        else:
            planned.append(PlannedChecker(name, config.timeout_s, tuple(config.depends_on)))
    # Drop checkers whose dependencies are not planned, repeating until nothing changes.
    changed = True
    while changed:
        names = {checker.name for checker in planned}
        kept = [c for c in planned if all(dep in names for dep in c.depends_on)]
        changed = len(kept) != len(planned)
        for checker in planned:
            if checker not in kept:
                skipped[checker.name] = "dependency_not_planned"
        planned = kept
    return Plan(kind=kind, checkers=tuple(planned), skipped=skipped)
