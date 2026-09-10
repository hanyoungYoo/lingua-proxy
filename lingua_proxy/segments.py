"""Mask technical content so a translation model cannot damage it.

Fenced code, inline code, URLs, paths, markup tags and JSON blobs are replaced
with ``<lpN/>`` placeholders before translation and restored afterwards. The
tag shape is deliberate: language models preserve XML-ish tags far more
reliably than exotic bracket glyphs, and when they do mangle one the damage is
predictable enough to repair (see :func:`normalize_placeholders`).

Two separate outputs matter:

* ``masked`` -- prose with placeholders, what gets sent to the translator.
* ``prose``  -- placeholders removed entirely, what gets sent to the language
  detector, so a Korean sentence wrapped around a big code block is not
  misread as English by the code's identifiers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

PLACEHOLDER_RE = re.compile(r"<lp(\d+)/>")

# Tolerates the ways a small model mangles a placeholder: swapped brackets,
# inserted spaces, uppercase, or a dropped slash.
_MANGLED_RE = re.compile(
    r"[\[⟦【〔<]\s*lp\s*(\d+)\s*/?\s*[\]⟧】〕>]",
    re.IGNORECASE,
)

_FENCE_RE = re.compile(
    r"(?P<fence>```|~~~)[^\n]*\n.*?(?P=fence)|(?P<open>```|~~~)[^\n]*\n.*", re.DOTALL
)
_INLINE_CODE_RE = re.compile(r"``.+?``|`[^`\n]+`", re.DOTALL)
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_TAG_RE = re.compile(r"</?[A-Za-z][\w.-]*(?:\s[^<>]*)?/?>")
_AT_MENTION_RE = re.compile(r"@[\w./~-]*/[\w./-]+|@[\w-]+\.[A-Za-z0-9]+")
_PATH_RE = re.compile(r"(?:~|\.{1,2})?/[\w.\-/]+|\b[\w.-]+\.[A-Za-z0-9]{1,6}\b")

# Extensions that make a bare token a filename rather than a sentence ending.
_KNOWN_EXTENSIONS = {
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "json",
    "toml",
    "yaml",
    "yml",
    "md",
    "txt",
    "sh",
    "rs",
    "go",
    "java",
    "rb",
    "c",
    "h",
    "cpp",
    "hpp",
    "css",
    "html",
    "sql",
    "ini",
    "cfg",
    "env",
    "lock",
    "csv",
    "tsv",
    "xml",
    "jsonl",
}

# Trailing characters that belong to the sentence, not to a URL or path.
_TRAILING = ".,;:!?)]}\"'"


@dataclass
class Masked:
    """Result of masking one piece of text."""

    masked: str
    placeholders: dict[str, str]
    prose: str


def _json_spans(text: str) -> list[tuple[int, int]]:
    """Find balanced ``{...}`` / ``[...]`` spans that actually parse as JSON."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        closing = "}" if ch == "{" else "]"
        depth = 0
        for j in range(i, len(text)):
            if text[j] == ch:
                depth += 1
            elif text[j] == closing:
                depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    try:
                        json.loads(candidate)
                    except ValueError:
                        break
                    spans.append((i, j + 1))
                    i = j
                    break
        i += 1
    return spans


def mask(text: str) -> Masked:
    """Replace technical spans with ``<lpN/>`` placeholders."""
    placeholders: dict[str, str] = {}
    counter = 0
    protected: list[tuple[int, int, str]] = []

    def claim(start: int, end: int, value: str) -> None:
        for pstart, pend, _ in protected:
            if start < pend and pstart < end:  # overlaps something already claimed
                return
        protected.append((start, end, value))

    # Order matters: the largest, least ambiguous spans are claimed first.
    for match in _FENCE_RE.finditer(text):
        claim(match.start(), match.end(), match.group())
    for match in _SYSTEM_REMINDER_RE.finditer(text):
        claim(match.start(), match.end(), match.group())
    for match in _INLINE_CODE_RE.finditer(text):
        claim(match.start(), match.end(), match.group())
    for start, end in _json_spans(text):
        claim(start, end, text[start:end])
    for match in _URL_RE.finditer(text):
        value = match.group().rstrip(_TRAILING)
        claim(match.start(), match.start() + len(value), value)
    for match in _TAG_RE.finditer(text):
        claim(match.start(), match.end(), match.group())
    for match in _AT_MENTION_RE.finditer(text):
        value = match.group().rstrip(_TRAILING)
        claim(match.start(), match.start() + len(value), value)
    for match in _PATH_RE.finditer(text):
        value = match.group().rstrip(_TRAILING)
        if "/" not in value:
            # A bare token only counts as a filename with a known extension,
            # so ordinary prose ending in a period is left alone.
            suffix = value.rsplit(".", 1)[-1].lower()
            if suffix not in _KNOWN_EXTENSIONS:
                continue
        claim(match.start(), match.start() + len(value), value)

    protected.sort(key=lambda item: item[0])

    out: list[str] = []
    cursor = 0
    for start, end, value in protected:
        out.append(text[cursor:start])
        token = f"<lp{counter}/>"
        counter += 1
        placeholders[token] = value
        out.append(token)
        cursor = end
    out.append(text[cursor:])

    masked = "".join(out)
    prose = PLACEHOLDER_RE.sub(" ", masked)
    return Masked(masked=masked, placeholders=placeholders, prose=prose)


def restore(text: str, placeholders: dict[str, str]) -> str:
    """Put the original technical spans back."""
    for token, value in placeholders.items():
        text = text.replace(token, value)
    return text


def normalize_placeholders(text: str) -> str:
    """Repair placeholders a model mangled into another bracket style."""
    return _MANGLED_RE.sub(lambda m: f"<lp{int(m.group(1))}/>", text)


def validate(source_masked: str, translated: str) -> set[str]:
    """Return placeholders that did not survive translation intact.

    A placeholder counts as damaged if it is missing or repeated: both would
    corrupt the restored text.
    """
    expected = PLACEHOLDER_RE.findall(source_masked)
    got = PLACEHOLDER_RE.findall(translated)
    damaged = set()
    for pid in expected:
        if got.count(pid) != expected.count(pid):
            damaged.add(f"<lp{pid}/>")
    for pid in got:
        if pid not in expected:
            damaged.add(f"<lp{pid}/>")
    return damaged
