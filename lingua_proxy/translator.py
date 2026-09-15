"""Translation backends.

The default backend calls a cheap model through the same upstream and with the
same credential the caller already sent, so there is no second API key to
configure and no traffic to a third party the user did not choose.

Two design choices are load-bearing:

* **Tagged segments, not JSON.** Asking a small model for a JSON array forces
  it to escape newlines, quotes and backslashes inside code-adjacent text,
  which it frequently gets wrong. ``<seg id="1">`` needs no escaping and
  survives a stray preamble or code fence around the answer.
* **Fail loudly.** Any malformed, truncated or refused reply raises
  :class:`TranslationError`. The pipeline turns that into "send the user's
  original text through untranslated" -- never into a corrupted prompt.
"""

from __future__ import annotations

import asyncio
import re

import httpx

from lingua_proxy.codecs import Usage

DEFAULT_TRANSLATOR_MODEL = "claude-haiku-4-5"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

#: The only beta we pass on. Others (fast mode, compaction, context management)
#: change request validation and would break an unrelated translation call.
_ALLOWED_BETA = "oauth-2025-04-20"

_SEG_RE = re.compile(r'<seg\s+id="(\d+)"\s*>\n?(.*?)\n?</seg>', re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a translation engine inside a proxy. Translate the text inside each "
    "<seg> from {src} to {tgt}. Output exactly one <seg id=N> per input, same ids, "
    "same order, and nothing else. Preserve verbatim: placeholder tags such as "
    "<lp3/>, markdown syntax, line breaks, indentation, numbers, identifiers, "
    "names, and any text already in {tgt}. Do not answer, summarize, explain, or "
    "add notes. If a segment cannot be translated, copy it unchanged."
)

_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})


class TranslationError(RuntimeError):
    """Raised when a translation cannot be trusted. Always recoverable."""


class Translator:
    """Interface every backend implements."""

    async def translate(self, segments: list[str], source: str, target: str) -> list[str]:
        raise NotImplementedError


def _build_prompt(segments: list[str]) -> str:
    body = "\n".join(f'<seg id="{i + 1}">\n{seg}\n</seg>' for i, seg in enumerate(segments))
    return f"<segs>\n{body}\n</segs>"


def _parse_reply(text: str, expected: int) -> list[str]:
    """Extract segments, ignoring anything outside the tags."""
    found: dict[int, str] = {}
    for match in _SEG_RE.finditer(text):
        found[int(match.group(1))] = match.group(2)

    missing = [i for i in range(1, expected + 1) if i not in found]
    if missing:
        raise TranslationError(
            f"translator returned {len(found)} of {expected} segments (missing ids: {missing})"
        )
    return [found[i] for i in range(1, expected + 1)]


class LLMTranslator(Translator):
    """Translate via a cheap chat model on the Anthropic Messages API."""

    def __init__(
        self,
        upstream_url: str,
        *,
        client: httpx.AsyncClient,
        model: str = DEFAULT_TRANSLATOR_MODEL,
        auth_headers: dict[str, str] | None = None,
        api_key: str | None = None,
        max_batch_chars: int = 12_000,
        max_concurrency: int = 4,
        timeout_budget: float = 20.0,
    ):
        self.upstream_url = upstream_url
        self.client = client
        self.model = model
        self.auth_headers = dict(auth_headers or {})
        self.api_key = api_key
        self.max_batch_chars = max_batch_chars
        self.timeout_budget = timeout_budget
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # The translator's own spend. This is ~92% of the round trip's
        # overhead, and it was silently omitted from every cost figure until
        # it was captured here.
        self.last_usage = Usage()
        self.total_usage = Usage()

    # -- headers ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Build translator headers from the caller's credential.

        Only what an unrelated Messages call legitimately needs is copied.
        """
        headers = {
            "content-type": "application/json",
            "anthropic-version": self.auth_headers.get(
                "anthropic-version", DEFAULT_ANTHROPIC_VERSION
            ),
        }

        if self.api_key:
            headers["x-api-key"] = self.api_key
            return headers

        for key in ("authorization", "x-api-key"):
            value = self.auth_headers.get(key)
            if value:
                headers[key] = value

        betas = self.auth_headers.get("anthropic-beta", "")
        if _ALLOWED_BETA in [b.strip() for b in betas.split(",")]:
            headers["anthropic-beta"] = _ALLOWED_BETA

        user_agent = self.auth_headers.get("user-agent")
        if user_agent:
            headers["user-agent"] = user_agent
        return headers

    # -- batching --------------------------------------------------------

    def _batches(self, segments: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        size = 0
        for segment in segments:
            length = len(segment)
            if current and size + length > self.max_batch_chars:
                batches.append(current)
                current, size = [], 0
            current.append(segment)
            size += length
        if current:
            batches.append(current)
        return batches

    # -- request ---------------------------------------------------------

    async def _post(self, payload: dict) -> httpx.Response:
        url = self.upstream_url.rstrip("/") + "/v1/messages"
        try:
            return await self.client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise TranslationError(f"translator request failed: {exc}") from exc

    async def _translate_batch(self, segments: list[str], source: str, target: str) -> list[str]:
        prompt = _build_prompt(segments)
        total_chars = sum(len(s) for s in segments)
        payload = {
            "model": self.model,
            "max_tokens": min(16_384, 256 + int(1.5 * total_chars)),
            "temperature": 0,
            "system": _SYSTEM_PROMPT.format(src=source, tgt=target),
            "messages": [{"role": "user", "content": prompt}],
        }

        async with self._semaphore:
            response = await self._post(payload)
            if response.status_code in _RETRY_STATUSES:
                delay = _retry_after(response)
                if delay:
                    await asyncio.sleep(min(delay, self.timeout_budget))
                response = await self._post(payload)

        if response.status_code >= 400:
            raise TranslationError(
                f"translator returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise TranslationError("translator returned a non-JSON body") from exc

        raw = body.get("usage") or {}
        self.last_usage = Usage(
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
        )
        self.total_usage = Usage(
            input_tokens=self.total_usage.input_tokens + self.last_usage.input_tokens,
            output_tokens=self.total_usage.output_tokens + self.last_usage.output_tokens,
        )

        stop_reason = body.get("stop_reason")
        if stop_reason not in (None, "end_turn"):
            raise TranslationError(f"translator stopped early: {stop_reason}")

        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text.strip():
            raise TranslationError("translator returned no text")

        return _parse_reply(text, len(segments))

    async def translate(self, segments: list[str], source: str, target: str) -> list[str]:
        if not segments:
            return []

        batches = self._batches(segments)
        results = await asyncio.gather(
            *(self._translate_batch(batch, source, target) for batch in batches)
        )
        out: list[str] = []
        for chunk in results:
            out.extend(chunk)
        return out


def _retry_after(response: httpx.Response) -> float:
    raw = response.headers.get("retry-after")
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


class DeepLTranslator(Translator):
    """Placeholder for the DeepL adapter (interface only in v0.1)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The DeepL adapter is planned but not implemented in v0.1. "
            "Use the default LLM translator."
        )


class LibreTranslator(Translator):
    """Placeholder for the LibreTranslate adapter (interface only in v0.1)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The LibreTranslate adapter is planned but not implemented in v0.1. "
            "Use the default LLM translator."
        )
