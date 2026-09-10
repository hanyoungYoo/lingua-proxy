"""Server-sent event parsing and re-emission.

Translating a streamed reply means taking events apart and putting them back
together. Two rules keep that safe:

* **Only text is rebuilt.** Every other event -- ``message_start``, tool-call
  deltas, reasoning with its cryptographic signature, usage, errors -- is
  forwarded as the exact bytes the upstream sent. Re-serializing JSON can
  reorder keys or change escaping, and a signature that no longer matches is
  rejected downstream.
* **Never go silent.** Clients abort a stream that produces no bytes for long
  enough, so keep-alive pings are emitted while a translation is in flight.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

#: Split translated text into deltas at most this large, on line boundaries,
#: so a long answer renders progressively instead of in one jump.
_MAX_DELTA_CHARS = 4096

PING_FRAME = b'event: ping\ndata: {"type":"ping"}\n\n'
OPENAI_KEEPALIVE = b": keepalive\n\n"


@dataclass
class SSEEvent:
    """One parsed server-sent event, with the bytes it arrived as."""

    name: str
    data: dict[str, Any]
    raw: bytes
    is_done: bool = False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SSEEvent(name={self.name!r}, type={self.data.get('type')!r})"


def _parse_frame(frame: bytes) -> SSEEvent | None:
    text = frame.decode("utf-8", errors="replace")
    name = ""
    data_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        # Comment lines (":" prefix) are keep-alives and carry no data.

    payload = "\n".join(data_lines).strip()
    if not payload:
        return None

    if payload == "[DONE]":
        return SSEEvent(name=name or "done", data={}, raw=frame, is_done=True)

    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    return SSEEvent(name=name or str(data.get("type", "")), data=data, raw=frame)


async def iter_sse_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[SSEEvent]:
    """Parse a byte stream into events, tolerating arbitrary chunk splits.

    Buffering as bytes (not text) matters: a multi-byte character can be split
    across TCP reads, and decoding each chunk separately would corrupt it.
    """
    buffer = bytearray()

    async for chunk in chunks:
        buffer.extend(chunk)
        while True:
            index = buffer.find(b"\n\n")
            crlf = buffer.find(b"\r\n\r\n")
            if crlf != -1 and (index == -1 or crlf < index):
                index, width = crlf, 4
            elif index != -1:
                width = 2
            else:
                break

            frame = bytes(buffer[: index + width])
            del buffer[: index + width]
            event = _parse_frame(frame)
            if event is not None:
                yield event

    if buffer.strip():
        event = _parse_frame(bytes(buffer))
        if event is not None:
            yield event


def _frame(name: str, data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n".encode()


def _split_text(text: str, limit: int = _MAX_DELTA_CHARS) -> list[str]:
    """Split on line boundaries so partial output still reads naturally."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > limit:
            parts.append(current)
            current = ""
        current += line
    if current:
        parts.append(current)
    return parts or [text]


def render_anthropic_text_block(index: int, text: str) -> list[bytes]:
    """Emit a complete, valid text block for the Anthropic Messages stream."""
    frames = [
        _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        )
    ]
    for part in _split_text(text) if text else [""]:
        if not part:
            continue
        frames.append(
            _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": part},
                },
            )
        )
    frames.append(_frame("content_block_stop", {"type": "content_block_stop", "index": index}))
    return frames


def render_openai_content_chunk(template: dict, index: int, text: str) -> list[bytes]:
    """Emit translated content chunks carrying the upstream's own identity."""
    frames = []
    for part in _split_text(text):
        payload = {
            "id": template.get("id", ""),
            "object": "chat.completion.chunk",
            "created": template.get("created", 0),
            "model": template.get("model", ""),
            "choices": [
                {
                    "index": index,
                    "delta": {"content": part},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        if "system_fingerprint" in template:
            payload["system_fingerprint"] = template["system_fingerprint"]
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        frames.append(f"data: {body}\n\n".encode())
    return frames


async def with_keepalive(
    work: Awaitable[list[bytes]],
    *,
    interval: float,
    frame: bytes = PING_FRAME,
) -> AsyncIterator[bytes]:
    """Emit keep-alives until ``work`` finishes, then its output.

    Without this a slow translation looks like a dead connection and the
    client gives up mid-answer.
    """
    task = asyncio.ensure_future(work)
    while True:
        done, _ = await asyncio.wait({task}, timeout=interval)
        if done:
            break
        yield frame

    for chunk in await task:
        yield chunk


async def translate_anthropic_stream(
    events: AsyncIterator[SSEEvent],
    translate: Callable[[list[str]], Awaitable[list[str]]],
    *,
    ping_interval: float = 15.0,
) -> AsyncIterator[bytes]:
    """Relay an Anthropic stream, translating only its text blocks.

    Text blocks are held until ``content_block_stop`` so the whole block can be
    translated as one unit; every other event is forwarded verbatim as soon as
    it arrives, which keeps tool calls and reasoning streaming at full speed.
    """
    pending_index: int | None = None
    pending_parts: list[str] = []
    saw_stop = False

    async for event in events:
        etype = event.data.get("type")

        if etype == "content_block_start":
            block = event.data.get("content_block") or {}
            if block.get("type") == "text":
                pending_index = event.data.get("index", 0)
                pending_parts = []
                continue  # withheld until the block closes
            yield event.raw
            continue

        if etype == "content_block_delta" and pending_index is not None:
            if event.data.get("index") == pending_index:
                delta = event.data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    pending_parts.append(delta.get("text", ""))
                    continue
            yield event.raw
            continue

        if etype == "content_block_stop" and pending_index is not None:
            if event.data.get("index") == pending_index:
                index, original = pending_index, "".join(pending_parts)
                pending_index, pending_parts = None, []

                async def work(text: str = original, at: int = index) -> list[bytes]:
                    # Both values are bound as defaults: the loop reassigns
                    # them before this coroutine actually runs.
                    try:
                        (translated,) = await translate([text])
                    except Exception:
                        # Better an untranslated answer than none at all.
                        translated = text
                    return render_anthropic_text_block(at, translated)

                async for chunk in with_keepalive(work(), interval=ping_interval):
                    yield chunk
                continue
            yield event.raw
            continue

        if etype == "message_stop":
            saw_stop = True
        yield event.raw

    if not saw_stop:
        # The upstream vanished mid-answer. Say so in the stream's own
        # language so the client can treat it as a retryable failure.
        yield _frame(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "lingua-proxy: upstream stream ended before message_stop",
                },
            },
        )


async def translate_openai_stream(
    events: AsyncIterator[SSEEvent],
    translate: Callable[[list[str]], Awaitable[list[str]]],
    *,
    ping_interval: float = 15.0,
) -> AsyncIterator[bytes]:
    """Relay an OpenAI chat stream, translating accumulated content."""
    template: dict[str, Any] = {}
    buffered: dict[int, list[str]] = {}
    tail: list[bytes] = []

    async def flush() -> list[bytes]:
        out: list[bytes] = []
        for index, parts in buffered.items():
            text = "".join(parts)
            if not text:
                continue
            try:
                (translated,) = await translate([text])
            except Exception:
                translated = text
            out.extend(render_openai_content_chunk(template, index, translated))
        return out

    async for event in events:
        if event.is_done:
            tail.append(event.raw)
            continue

        for key in ("id", "created", "model", "system_fingerprint"):
            if key in event.data and key not in template:
                template[key] = event.data[key]

        choices = event.data.get("choices")
        if not isinstance(choices, list) or not choices:
            # Usage or metadata chunk: emit after the translated content.
            tail.append(event.raw)
            continue

        held = False
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str) and content:
                buffered.setdefault(choice.get("index", 0), []).append(content)
                held = True

        if choice_finished(choices):
            tail.append(strip_content(event))
            continue
        if not held:
            yield event.raw

    async for chunk in with_keepalive(flush(), interval=ping_interval, frame=OPENAI_KEEPALIVE):
        yield chunk
    for chunk in tail:
        yield chunk


def choice_finished(choices: list) -> bool:
    return any(
        isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices
    )


def strip_content(event: SSEEvent) -> bytes:
    """Re-emit a finish chunk without its text, which we already translated."""
    data = json.loads(json.dumps(event.data))
    for choice in data.get("choices", []):
        if isinstance(choice, dict) and isinstance(choice.get("delta"), dict):
            choice["delta"].pop("content", None)
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n".encode()
