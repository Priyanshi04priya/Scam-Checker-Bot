"""Link inspection: shorteners, brand look-alikes and mixed-script (homoglyph) domains."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from rapidfuzz.distance import Levenshtein

from scambot.core.config import FuzzyThresholds
from scambot.core.domains import domain_label, registrable_domain, url_host

# Cyrillic, Greek and IPA letters that look like Latin ones (written as escapes so the
# source itself has no look-alike characters).
CONFUSABLES: Final = str.maketrans(
    {
        "\u0430": "a",  # cyrillic small letter a
        "\u0435": "e",  # cyrillic small letter ie
        "\u043e": "o",  # cyrillic small letter o
        "\u0440": "p",  # cyrillic small letter er
        "\u0441": "c",  # cyrillic small letter es
        "\u0443": "y",  # cyrillic small letter u
        "\u0445": "x",  # cyrillic small letter ha
        "\u0456": "i",  # cyrillic small letter byelorussian-ukrainian i
        "\u0458": "j",  # cyrillic small letter je
        "\u0455": "s",  # cyrillic small letter dze
        "\u0501": "d",  # cyrillic small letter komi de
        "\u0261": "g",  # latin small letter script g
        "\u03bf": "o",  # greek small letter omicron
        "\u03bd": "v",  # greek small letter nu
        "\u03b1": "a",  # greek small letter alpha
    }
)
_LOOKALIKE_PAIRS: Final = (("rn", "m"), ("vv", "w"), ("0", "o"), ("1", "l"), ("3", "e"), ("5", "s"))
_TOKEN_SPLIT_RE: Final = re.compile(r"[^0-9a-z\u0080-￿]+")

SHORTENER_FLAG: Final = "shortener"
PUNYCODE_FLAG: Final = "punycode"
_NEUTRAL_FLAGS: Final = frozenset({SHORTENER_FLAG, PUNYCODE_FLAG})


def skeleton(text: str) -> str:
    """A canonical form in which look-alike characters compare equal."""
    result = text.translate(CONFUSABLES).lower()
    for source, target in _LOOKALIKE_PAIRS:
        result = result.replace(source, target)
    return result


@dataclass(frozen=True)
class UrlFinding:
    url: str
    host: str
    flags: list[str] = field(default_factory=list)
    brand: str | None = None

    @property
    def is_shortener(self) -> bool:
        return SHORTENER_FLAG in self.flags

    @property
    def is_lookalike(self) -> bool:
        """A brand, look-alike, homoglyph or invalid-IDN flag (punycode alone is not)."""
        return any(flag not in _NEUTRAL_FLAGS for flag in self.flags)


def _decode_idn(host: str) -> tuple[str, bool]:
    """The Unicode form of ``host`` and whether every punycode label decoded."""
    labels: list[str] = []
    valid = True
    for label in host.split("."):
        if label.startswith("xn--"):
            try:
                labels.append(label.encode("ascii").decode("idna"))
            except UnicodeError:
                valid = False
                labels.append(label)
        else:
            labels.append(label)
    return ".".join(labels), valid


def _is_official(host: str, domains: Sequence[str]) -> bool:
    return any(host == d or host.endswith(f".{d}") for d in domains)


def _is_lookalike(token: str, brand: str, fuzzy: FuzzyThresholds) -> bool:
    if len(token) < fuzzy.min_token_len:
        return False
    shape = skeleton(token)
    if shape == brand != token:
        return True
    # Edit distance on very short brand names ("sbi", "jio") would match ordinary words.
    if len(brand) < fuzzy.min_token_len:
        return False
    limit = (
        fuzzy.max_distance_short
        if len(brand) <= fuzzy.short_brand_max_len
        else fuzzy.max_distance_long
    )
    return 1 <= Levenshtein.distance(shape, brand) <= limit


def _brand_flags(
    host: str, label: str, brands: Mapping[str, Sequence[str]], fuzzy: FuzzyThresholds
) -> tuple[list[str], str | None]:
    if any(_is_official(host, domains) for domains in brands.values()):
        return [], None
    flags: list[str] = []
    found: str | None = None
    tokens = [t for t in _TOKEN_SPLIT_RE.split(label.lower()) if t]
    dotted = f".{host}."
    for brand, domains in brands.items():
        before = len(flags)
        in_domain = brand in tokens or (len(brand) >= 5 and brand in label)
        if in_domain:
            flags.append(f"brand_in_domain:{brand}")
        if any(f".{d}." in dotted for d in domains):
            flags.append(f"brand_in_subdomain:{brand}")
        if not in_domain and any(_is_lookalike(token, brand, fuzzy) for token in tokens):
            flags.append(f"lookalike:{brand}")
        if len(flags) > before and found is None:
            found = brand
    return flags, found


def _script_flags(host: str) -> list[str]:
    if not any(label.startswith("xn--") for label in host.split(".")):
        return []
    flags = [PUNYCODE_FLAG]
    decoded, valid = _decode_idn(host)
    if not valid:
        return [*flags, "invalid_idn"]
    letters = [c for c in decoded if c.isalpha()]
    has_ascii = any(c.isascii() for c in letters)
    has_other = any(not c.isascii() for c in letters)
    shape = skeleton(decoded)
    if (has_ascii and has_other) or (shape.isascii() and shape != decoded):
        flags.append("homoglyph")
    return flags


def inspect_url(
    url: str,
    brands: Mapping[str, Sequence[str]],
    shorteners: frozenset[str],
    fuzzy: FuzzyThresholds,
) -> UrlFinding:
    """Flags for one link; an official brand domain gets no brand flags."""
    host = url_host(url) or ""
    flags: list[str] = []
    if host and registrable_domain(host) in shorteners:
        flags.append(SHORTENER_FLAG)
    decoded, _ = _decode_idn(host)
    label = domain_label(decoded) if decoded else ""
    brand_flags, brand = _brand_flags(decoded, label, brands, fuzzy) if label else ([], None)
    flags += brand_flags + _script_flags(host)
    return UrlFinding(url=url, host=host, flags=flags, brand=brand)
