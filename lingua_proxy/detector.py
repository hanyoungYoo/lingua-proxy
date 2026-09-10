"""Offline language detection.

Uses ``lingua``: deterministic, no network call, no per-request cost. Two
guards sit around it because statistical detection is weakest on exactly the
input a coding proxy sees most -- short fragments and symbol-heavy text.

1. A minimum length, below which we never translate.
2. A script-share check, so an English sentence quoting a Korean word is not
   flipped to Korean by a handful of characters.

Every ambiguous case resolves to "do not translate". A missed translation
costs a saving; a wrong one corrupts the user's prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lingua import Language, LanguageDetectorBuilder

from lingua_proxy.config import DEFAULT_LANGUAGES

# Scripts whose tokenization penalty makes translation worth it.
_NON_LATIN_RANGES = (
    (0xAC00, 0xD7A3),  # Hangul syllables
    (0x1100, 0x11FF),  # Hangul jamo
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0x0600, 0x06FF),  # Arabic
    (0x0400, 0x04FF),  # Cyrillic
    (0x0900, 0x097F),  # Devanagari
    (0x0E00, 0x0E7F),  # Thai
)

_LATIN_RE = re.compile(r"[A-Za-z]")

#: Below this share of non-Latin letters, a text with substantial Latin content
#: is treated as English without consulting the statistical model.
_NON_LATIN_MIN_SHARE = 0.15
_LATIN_LETTERS_FOR_GUARD = 20


@dataclass(frozen=True)
class Decision:
    """Outcome of detecting one piece of text."""

    lang: str | None
    confidence: float
    translate: bool


def _is_non_latin(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _NON_LATIN_RANGES)


def non_latin_share(text: str) -> float:
    """Share of letter-like characters that belong to a non-Latin script."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if _is_non_latin(ch)) / len(letters)


class Detector:
    """Wraps a preloaded ``lingua`` detector with fail-safe guards."""

    def __init__(
        self,
        languages: tuple[str, ...] = DEFAULT_LANGUAGES,
        *,
        min_chars: int = 20,
        min_confidence: float = 0.6,
        minimum_relative_distance: float = 0.2,
    ):
        self.min_chars = min_chars
        self.min_confidence = min_confidence

        by_code = {}
        for language in Language.all():
            iso = language.iso_code_639_1.name.lower()
            by_code[iso] = language

        selected = [by_code[code.lower()] for code in languages if code.lower() in by_code]
        if by_code["en"] not in selected:
            selected.append(by_code["en"])
        # lingua needs at least two languages to discriminate between.
        if len(selected) < 2:
            selected.append(by_code["ko"])

        self.languages = tuple(sorted({lang.iso_code_639_1.name.lower() for lang in selected}))
        self._detector = (
            LanguageDetectorBuilder.from_languages(*selected)
            .with_minimum_relative_distance(minimum_relative_distance)
            .with_preloaded_language_models()
            .build()
        )

    def detect(self, text: str) -> Decision:
        """Decide whether ``text`` should be translated, and from what."""
        stripped = (text or "").strip()
        if len(stripped) < self.min_chars:
            return Decision(lang=None, confidence=0.0, translate=False)

        share = non_latin_share(stripped)
        latin_letters = len(_LATIN_RE.findall(stripped))

        if share == 0.0:
            # No non-Latin script at all: nothing here is worth translating.
            return Decision(lang="en", confidence=1.0, translate=False)

        if share < _NON_LATIN_MIN_SHARE and latin_letters >= _LATIN_LETTERS_FOR_GUARD:
            # Mostly Latin with a sprinkling of other script: treat as English.
            return Decision(lang="en", confidence=1.0 - share, translate=False)

        language = self._detector.detect_language_of(stripped)
        if language is None:
            return Decision(lang=None, confidence=0.0, translate=False)

        code = language.iso_code_639_1.name.lower()
        confidence = self._detector.compute_language_confidence(stripped, language)

        # Whole-text detection is dominated by whichever clause comes first, so
        # a prompt that opens in English and continues in Korean reads as
        # English. The script share already told us a substantial part of this
        # text is not Latin, so ask which non-Latin language it actually is.
        if code == "en" and share >= _NON_LATIN_MIN_SHARE:
            code, confidence = self._dominant_non_latin(stripped)
            if code is None:
                return Decision(lang="en", confidence=1.0, translate=False)

        if code == "en":
            return Decision(lang="en", confidence=confidence, translate=False)
        if confidence < self.min_confidence:
            return Decision(lang=code, confidence=confidence, translate=False)
        return Decision(lang=code, confidence=confidence, translate=True)

    def _dominant_non_latin(self, text: str) -> tuple[str | None, float]:
        """Find the non-Latin language covering the most characters.

        Uses ``detect_multiple_languages_of`` so each clause is judged on its
        own, then picks the non-Latin language with the widest coverage.
        """
        by_lang: dict[str, int] = {}
        try:
            segments = self._detector.detect_multiple_languages_of(text)
        except Exception:  # pragma: no cover - defensive; lingua marks this experimental
            return None, 0.0

        for result in segments:
            span = text[result.start_index : result.end_index]
            if non_latin_share(span) < _NON_LATIN_MIN_SHARE:
                continue
            code = result.language.iso_code_639_1.name.lower()
            if code == "en":
                continue
            by_lang[code] = by_lang.get(code, 0) + len(span.strip())

        if not by_lang:
            return None, 0.0

        best = max(by_lang, key=lambda key: by_lang[key])
        coverage = by_lang[best] / max(len(text.strip()), 1)
        return best, min(1.0, coverage + 0.5)
