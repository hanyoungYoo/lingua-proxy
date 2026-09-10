"""Translate in, forward, translate out.

This module owns the request flow and every decision about *whether* to
translate. Two invariants govern it:

**Fail open.** Any failure -- detection uncertainty, a translator error, a
damaged placeholder -- results in the user's original text being forwarded
untranslated. A missed saving is a rounding error; a corrupted prompt is not.

**Byte stability.** Text already translated in an earlier turn is reproduced
from the memo, never re-translated. Agentic clients resend the whole history
each turn, so anything else would break the upstream prompt cache and cost
more than it saves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fnmatch import fnmatch

from lingua_proxy.codecs import Codec, Ref
from lingua_proxy.detector import Detector
from lingua_proxy.memo import Memo
from lingua_proxy.segments import mask, normalize_placeholders, restore, validate

ENGLISH = "en"


class TranslationSkipped(Exception):
    """Internal signal: forward the original request unchanged."""


@dataclass
class Outcome:
    """What the pipeline did, for the cost log."""

    translated: bool = False
    source_lang: str | None = None
    skipped_reason: str | None = None
    fallback_reason: str | None = None
    translator_calls: int = 0
    memo_hits: int = 0
    memo_miss_assistant: int = 0
    original_body: dict | None = None
    forwarded_body: dict | None = None
    english_texts: list[str] = field(default_factory=list)
    translated_texts: list[str] = field(default_factory=list)


class Pipeline:
    """Applies translation to one request/response pair."""

    def __init__(
        self,
        *,
        detector: Detector,
        translator,
        memo: Memo,
        skip_models: tuple[str, ...] = (),
        max_output_tokens: int = 0,
    ):
        self.detector = detector
        self.translator = translator
        self.memo = memo
        self.skip_models = skip_models
        self.max_output_tokens = max_output_tokens

    # -- decisions -------------------------------------------------------

    def should_skip(self, body: dict, codec: Codec, bypass: bool) -> str | None:
        """Reasons to forward without looking at the text at all."""
        if bypass:
            return "bypass_header"

        model = codec.model(body)
        if model and any(fnmatch(model, pattern) for pattern in self.skip_models):
            return "model"

        # A long reply costs more to translate than the request saves, because
        # the fee scales with the answer while the saving does not.
        if self.max_output_tokens:
            requested = codec.max_output_tokens(body)
            if requested and requested > self.max_output_tokens:
                return "long_output"

        return None

    @staticmethod
    def _has_prose(text: str) -> bool:
        """True when a segment contains words, not just protected spans.

        A block that is entirely a code fence or an injected
        ``<system-reminder>`` masks down to nothing, and sending that to a
        translation model wastes a call and risks damaging it.
        """
        return bool(mask(text).prose.strip())

    def _detect_ref(self, ref: Ref):
        """Detect the language of one segment, ignoring injected reminders."""
        masked = mask(ref.text)
        prose = masked.prose.strip()
        if not prose:
            return None
        return self.detector.detect(prose)

    def conversation_language(self, body: dict, codec: Codec) -> str | None:
        """Which language the user is speaking, or None to pass through.

        The latest turn decides when it contains prose. When it does not --
        a tool-result-only turn, or a bare "go ahead" -- earlier user turns
        are consulted through the memo so replies keep their language across
        the tool calls that dominate agent traffic.
        """
        for ref in codec.user_refs(body):
            decision = self._detect_ref(ref)
            if decision is None:
                continue
            if decision.translate and decision.lang:
                return decision.lang
            if decision.lang == ENGLISH:
                # An explicit English turn means the user switched.
                return None

        # No usable prose in this turn: inherit from history.
        for ref in reversed(codec.assistant_refs(body)):
            hit = self.memo.get(ref.text)
            if hit is not None and hit.source and hit.source != ENGLISH:
                return hit.source
        return None

    # -- translation helpers ---------------------------------------------

    async def _translate_texts(
        self, texts: list[str], source: str, target: str, outcome: Outcome
    ) -> dict[str, str]:
        """Translate a set of strings, protecting technical spans.

        Raises :class:`TranslationSkipped` if anything cannot be trusted.
        """
        if not texts:
            return {}

        masked_by_text = {text: mask(text) for text in texts}
        payload = [masked_by_text[text].masked for text in texts]

        try:
            outputs = await self.translator.translate(payload, source, target)
        except Exception as exc:  # noqa: BLE001 - any failure means "do not translate"
            outcome.fallback_reason = f"translator_error: {type(exc).__name__}"
            raise TranslationSkipped from exc

        outcome.translator_calls += 1

        if len(outputs) != len(texts):
            outcome.fallback_reason = "segment_count_mismatch"
            raise TranslationSkipped

        result: dict[str, str] = {}
        for text, produced in zip(texts, outputs, strict=True):
            masked = masked_by_text[text]
            repaired = normalize_placeholders(produced)
            if validate(masked.masked, repaired):
                outcome.fallback_reason = "placeholder_loss"
                raise TranslationSkipped
            result[text] = restore(repaired, masked.placeholders)
        return result

    # -- request ---------------------------------------------------------

    async def translate_request(
        self, body: dict, codec: Codec, source: str
    ) -> tuple[dict, Outcome]:
        """Rewrite a request into English."""
        outcome = Outcome(source_lang=source, original_body=body)

        pending: list[str] = []
        replacements: dict[str, str] = {}

        user_refs = [ref for ref in codec.user_refs(body) if self._has_prose(ref.text)]
        for ref in user_refs:
            hit = self.memo.get(ref.text)
            if hit is not None:
                replacements[ref.text] = hit.text
                outcome.memo_hits += 1
            elif ref.text not in pending:
                pending.append(ref.text)

        fresh = await self._translate_texts(pending, source, ENGLISH, outcome)
        for original, english in fresh.items():
            self.memo.put(original, english, source=source, target=ENGLISH, direction="req")
            replacements[original] = english

        updated = body
        for ref in user_refs:
            english = replacements.get(ref.text)
            if english is not None and english != ref.text:
                updated = codec.apply(updated, ref, english)

        # History must be reproduced exactly as it was sent before, or the
        # upstream prompt cache misses on every turn and we cost more than we
        # save. Earlier user turns come from the memo; they are never
        # re-translated, because a second translation could differ.
        for ref in codec.history_user_refs(updated):
            hit = self.memo.get(ref.text)
            if hit is not None and hit.target == ENGLISH:
                updated = codec.apply(updated, ref, hit.text)
                outcome.memo_hits += 1

        # Our translated replies are swapped back to the model's own English
        # so the upstream sees precisely what it produced.
        for ref in codec.assistant_refs(updated):
            hit = self.memo.get(ref.text)
            if hit is not None and hit.target == ENGLISH:
                updated = codec.apply(updated, ref, hit.text)
                outcome.memo_hits += 1
            else:
                outcome.memo_miss_assistant += 1

        outcome.translated = True
        outcome.forwarded_body = updated
        return updated, outcome

    # -- response --------------------------------------------------------

    async def translate_response(
        self, response: dict, codec: Codec, target: str, outcome: Outcome
    ) -> dict:
        """Rewrite a reply back into the user's language."""
        refs = codec.response_refs(response)
        if not refs:
            return response

        texts = [ref.text for ref in refs]
        try:
            produced = await self._translate_texts(texts, ENGLISH, target, outcome)
        except TranslationSkipped:
            # The user gets the English answer rather than no answer.
            return response

        updated = response
        for ref in refs:
            localized = produced.get(ref.text)
            if localized is None or localized == ref.text:
                continue
            self.memo.put(ref.text, localized, source=ENGLISH, target=target, direction="resp")
            outcome.english_texts.append(ref.text)
            outcome.translated_texts.append(localized)
            updated = codec.apply(updated, ref, localized)
        return updated

    async def translate_texts_for_stream(
        self, texts: list[str], target: str, outcome: Outcome
    ) -> dict[str, str]:
        """Translate streamed reply text, recording pairs in the memo.

        Used by the streaming relay, where blocks arrive one at a time rather
        than as a complete response document.
        """
        produced = await self._translate_texts(texts, ENGLISH, target, outcome)
        for english, localized in produced.items():
            self.memo.put(english, localized, source=ENGLISH, target=target, direction="resp")
            outcome.english_texts.append(english)
            outcome.translated_texts.append(localized)
        return produced


def parse_body(raw: bytes) -> dict | None:
    """Parse a request body, or None if it is not a JSON object."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
