"""Reply grounding audit: every URL, date and number in a reply must come from the evidence.

The automated pass flags a reply that states something no checker reported and that the user
did not send. ``sample_for_manual_review`` writes the stratified sample a person checks by hand.
"""

import csv
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from eval.tasks import ItemResult
from scambot.core.locales import Locales

_URL_RE: Final = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_TIME_RE: Final = re.compile(r"\b\d{1,2}:\d{2}\b")
_NUMBER_RE: Final = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ISO_RE: Final = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_RE: Final = re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b")
_YEAR_RE: Final = re.compile(r"\b(?:19|20)\d{2}\b")
_EN_MONTHS: Final = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]  # fmt: skip
_HI_INDEX: Final = {
    name: idx
    for idx, name in (
        (1, "जनवरी"),
        (2, "फ़रवरी"),
        (2, "फरवरी"),
        (3, "मार्च"),
        (4, "अप्रैल"),
        (5, "मई"),
        (6, "जून"),
        (7, "जुलाई"),
        (8, "अगस्त"),
        (9, "सितंबर"),
        (9, "सितम्बर"),
        (10, "अक्टूबर"),
        (11, "नवंबर"),
        (11, "नवम्बर"),
        (12, "दिसंबर"),
        (12, "दिसम्बर"),
    )
}
_HI_MONTHS: Final = list(_HI_INDEX)
_MONTH_NAMES: Final = "|".join(_EN_MONTHS + _HI_MONTHS)
_NAMED_DATE_RE: Final = re.compile(
    rf"\b(?:(\d{{1,2}})\s+)?({_MONTH_NAMES})\s+((?:19|20)\d{{2}})", re.IGNORECASE
)
_DIGITS: Final = str.maketrans("०१२३४५६७८९", "0123456789")

Date = tuple[int, int | None, int | None]


def normalise_digits(text: str) -> str:
    """Devanagari digits become ASCII digits."""
    return text.translate(_DIGITS)


def _month(name: str) -> int:
    lowered = name.lower()
    if lowered in _EN_MONTHS:
        return _EN_MONTHS.index(lowered) + 1
    return _HI_INDEX.get(name, 0)


def _url_key(url: str) -> str:
    rest = re.sub(r"^https?://", "", url.strip().rstrip(".,;)"), flags=re.IGNORECASE)
    rest = re.sub(r"^www\.", "", rest, flags=re.IGNORECASE).split("#")[0].split("?")[0]
    return rest.rstrip("/").lower()


def _dates_in(text: str) -> tuple[set[Date], str]:
    """Dates found in ``text``, and the text with them removed."""
    found: set[Date] = set()

    def iso(m: re.Match[str]) -> str:
        found.add((int(m[1]), int(m[2]), int(m[3])))
        return " "

    def dmy(m: re.Match[str]) -> str:
        found.add((int(m[3]), int(m[2]), int(m[1])))
        return " "

    def named(m: re.Match[str]) -> str:
        day = int(m[1]) if m[1] else None
        found.add((int(m[3]), _month(m[2]), day))
        return " "

    text = _ISO_RE.sub(iso, text)
    text = _DMY_RE.sub(dmy, text)
    text = _NAMED_DATE_RE.sub(named, text)
    for year in _YEAR_RE.findall(text):
        found.add((int(year), None, None))
    return found, _YEAR_RE.sub(" ", text)


def _expand(dates: Iterable[Date]) -> set[Date]:
    out: set[Date] = set()
    for year, month, day in dates:
        out |= {(year, month, day), (year, month, None), (year, None, None)}
    return out


def _numbers_in(text: str) -> set[str]:
    cleaned = _TIME_RE.sub(" ", text)
    return {m.replace(",", "").rstrip(".") for m in _NUMBER_RE.findall(cleaned)}


def extract_claims(reply_text: str) -> dict[str, set[Any]]:
    """URLs, dates and numbers a reply states (Devanagari digits are normalised)."""
    text = normalise_digits(reply_text)
    urls = {_url_key(u) for u in _URL_RE.findall(text)}
    without_urls = _URL_RE.sub(" ", text)
    dates, rest = _dates_in(without_urls)
    return {"urls": urls, "dates": dates, "numbers": _numbers_in(rest)}


def _walk(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v)
    elif value is not None:
        yield str(value)


def template_numbers(locales: Locales) -> set[str]:
    """Numbers that appear in the fixed locale strings (limits, retention days, ...)."""
    numbers: set[str] = set()
    for language in locales.languages:
        for text in _walk(locales.get(language).model_dump()):
            numbers |= _numbers_in(normalise_digits(text))
    return numbers


def allowed_values(result: ItemResult, user_text: str = "") -> dict[str, set[Any]]:
    """Everything the reply for ``result`` may state: evidence, details, score, the user's text."""
    texts: list[str] = [user_text]
    urls: set[str] = set()
    dates: set[Date] = set()
    for check in result.checks:
        for evidence in check.evidence:
            texts.append(evidence.summary)
            if evidence.url:
                urls.add(_url_key(evidence.url))
            if evidence.source:
                texts.append(evidence.source)
            if evidence.date:
                dates.add((evidence.date.year, evidence.date.month, evidence.date.day))
        texts += list(_walk(check.details))
        texts.append(json.dumps(check.details, ensure_ascii=False, default=str))
    numbers: set[str] = {str(result.score)} if result.score is not None else set()
    for text in texts:
        normal = normalise_digits(text)
        urls |= {_url_key(u) for u in _URL_RE.findall(normal)}
        found, rest = _dates_in(_URL_RE.sub(" ", normal))
        dates |= found
        numbers |= _numbers_in(rest) | _numbers_in(normal)
        numbers |= {re.sub(r"\D", "", normal)} if normal.strip().isdigit() else set()
    return {"urls": urls, "dates": _expand(dates), "numbers": numbers}


def audit_reply(result: ItemResult, user_text: str, static_numbers: set[str]) -> list[str]:
    """The URLs, dates and numbers in the reply that nothing supports (empty = grounded)."""
    claims = extract_claims(result.reply_text)
    allowed = allowed_values(result, user_text)
    bad: list[str] = []
    bad += [f"url:{u}" for u in sorted(claims["urls"] - allowed["urls"])]
    bad += [
        f"date:{d[0]}-{d[1] or 0:02d}-{d[2] or 0:02d}"
        for d in sorted(claims["dates"], key=str)
        if d not in allowed["dates"]
    ]
    bad += [
        f"number:{n}"
        for n in sorted(claims["numbers"])
        if n not in allowed["numbers"] and n not in static_numbers
    ]
    return bad


def run_audit(
    results: Sequence[ItemResult], texts: Mapping[str, str], locales: Locales
) -> list[dict[str, Any]]:
    """One row per replied result: ``item_id, language, reply_text, unsupported_items, pass``."""
    static = template_numbers(locales)
    rows: list[dict[str, Any]] = []
    for result in results:
        if not result.reply_text:
            continue
        bad = [] if result.cached else audit_reply(result, texts.get(result.item_id, ""), static)
        rows.append(
            {
                "item_id": result.item_id,
                "language": result.language,
                "reply_text": result.reply_text,
                "unsupported_items": "; ".join(bad),
                "pass": not bad,
            }
        )
    return rows


def audit_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    failing = sum(1 for r in rows if not r["pass"])
    return {"audited": len(rows), "failing": failing, "passing": len(rows) - failing}


def write_audit_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = ["item_id", "language", "reply_text", "unsupported_items", "pass"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_for_manual_review(
    results: Sequence[ItemResult], n: int = 100, seed: int = 1
) -> list[ItemResult]:
    """A reproducible sample spread evenly over (label, language) groups."""
    groups: dict[tuple[str, str], list[ItemResult]] = defaultdict(list)
    for r in sorted(results, key=lambda r: r.item_id):
        if r.reply_text:
            groups[(r.predicted_label or "none", r.language)].append(r)
    rng = random.Random(seed)  # noqa: S311 - a reproducible sample, not security
    for group in groups.values():
        rng.shuffle(group)
    picked: list[ItemResult] = []
    queues = [groups[k] for k in sorted(groups)]
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return sorted(picked, key=lambda r: r.item_id)


def write_manual_review(sample: Sequence[ItemResult], path: Path) -> None:
    fields = [
        "item_id",
        "language",
        "label",
        "reply_text",
        "checker_evidence",
        "supported(Y/N)",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in sample:
            evidence = " | ".join(
                f"{c.checker}: {e.summary}" + (f" <{e.url}>" if e.url else "")
                for c in r.checks
                for e in c.evidence
            )
            writer.writerow(
                {
                    "item_id": r.item_id,
                    "language": r.language,
                    "label": r.predicted_label,
                    "reply_text": r.reply_text,
                    "checker_evidence": evidence,
                    "supported(Y/N)": "",
                    "notes": "",
                }
            )
