"""Prompt files: a system section and a user template whose variables sit inside a data block.

Everything forwarded by a user is placed between ``<forwarded_content>`` tags, and the system
prompt tells the model to treat that block as data only. Values cannot close the block early.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scambot.core.settings import CONFIG_DIR

PROMPTS_DIR: Final = CONFIG_DIR / "prompts"

SYSTEM_MARKER: Final = "## system"
USER_MARKER: Final = "## user"
OPEN_TAG: Final = "<forwarded_content>"
CLOSE_TAG: Final = "</forwarded_content>"

_NAME_RE = re.compile(r"^[a-z_]+$")
_VARIABLE_RE = re.compile(r"\{\{([a-z_][a-z0-9_]*)\}\}")
_TAG_RE = re.compile(r"(?i)<\s*/?\s*forwarded_content\s*>")
TAG_REPLACEMENT: Final = "[tag removed]"


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    user: str
    variables: frozenset[str]


def _fail(path: Path, problem: str) -> ValueError:
    return ValueError(f"invalid prompt file {path.name}: {problem}")


def _parse(path: Path) -> Prompt:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines.count(SYSTEM_MARKER) != 1 or lines.count(USER_MARKER) != 1:
        raise _fail(path, f"needs exactly one {SYSTEM_MARKER!r} and one {USER_MARKER!r} line")
    if lines[0] != SYSTEM_MARKER:
        raise _fail(path, f"the first line must be {SYSTEM_MARKER!r}")
    split = lines.index(USER_MARKER)
    system = "\n".join(lines[1:split]).strip()
    user = "\n".join(lines[split + 1 :]).strip()
    if not system or not user:
        raise _fail(path, "both sections must be non-empty")
    if "{{" in system or "}}" in system:
        raise _fail(path, "the system section may not contain placeholders")
    if user.count(OPEN_TAG) != 1 or user.count(CLOSE_TAG) != 1:
        raise _fail(path, f"the user section needs exactly one {OPEN_TAG} ... {CLOSE_TAG} block")
    start = user.index(OPEN_TAG)
    end = user.index(CLOSE_TAG)
    if end < start:
        raise _fail(path, f"{CLOSE_TAG} comes before {OPEN_TAG}")
    variables: set[str] = set()
    for match in _VARIABLE_RE.finditer(user):
        if not start < match.start() < end:
            raise _fail(path, f"placeholder {match.group(0)} is outside the data block")
        variables.add(match.group(1))
    leftover = _VARIABLE_RE.sub("", user)
    if "{{" in leftover or "}}" in leftover:
        raise _fail(path, "malformed placeholder")
    return Prompt(path.stem, system, user, frozenset(variables))


def sanitise(value: str) -> str:
    """Stop a value from opening or closing the data block."""
    return _TAG_RE.sub(TAG_REPLACEMENT, value)


class PromptLibrary:
    """All loaded prompts by name."""

    def __init__(self, prompts: Mapping[str, Prompt]) -> None:
        self._prompts = dict(prompts)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._prompts)

    def render(self, name: str, variables: Mapping[str, str]) -> tuple[str, str]:
        """Return ``(system, user)`` with every placeholder filled in."""
        prompt = self._prompts.get(name)
        if prompt is None:
            msg = f"unknown prompt {name!r}"
            raise ValueError(msg)
        given = set(variables)
        if given != prompt.variables:
            missing = sorted(prompt.variables - given)
            extra = sorted(given - prompt.variables)
            msg = f"prompt {name!r}: missing variables {missing}, unexpected variables {extra}"
            raise ValueError(msg)
        user = _VARIABLE_RE.sub(lambda m: sanitise(variables[m.group(1)]), prompt.user)
        return prompt.system, user


def load_prompts(directory: Path = PROMPTS_DIR) -> PromptLibrary:
    """Load and validate every ``*.md`` prompt in ``directory``; the file stem is the name."""
    prompts: dict[str, Prompt] = {}
    for path in sorted(directory.glob("*.md")):
        if not _NAME_RE.match(path.stem):
            raise _fail(path, "prompt names may only use lower-case letters and underscores")
        prompts[path.stem] = _parse(path)
    return PromptLibrary(prompts)
