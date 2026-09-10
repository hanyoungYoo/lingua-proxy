"""Streaming: translate text blocks without breaking the event protocol.

Text blocks are buffered per block, translated, then re-emitted. Everything
else -- tool calls, reasoning, usage, errors -- passes through as raw bytes,
because those events carry signatures and IDs that must not be re-serialized.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lingua_proxy.streaming import (
    SSEEvent,
    iter_sse_events,
    render_anthropic_text_block,
    render_openai_content_chunk,
)


def frame(name: str, data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n".encode()


async def feed(*chunks: bytes):
    for chunk in chunks:
        yield chunk


# -- parser -------------------------------------------------------------


async def test_parses_events_split_on_blank_lines():
    stream = feed(
        frame("message_start", {"type": "message_start"}),
        frame("message_stop", {"type": "message_stop"}),
    )
    events = [e async for e in iter_sse_events(stream)]

    assert [e.name for e in events] == ["message_start", "message_stop"]
    assert events[0].data["type"] == "message_start"


async def test_parses_event_split_across_chunk_boundaries():
    whole = frame("message_start", {"type": "message_start"})
    stream = feed(whole[:10], whole[10:22], whole[22:])
    events = [e async for e in iter_sse_events(stream)]

    assert [e.name for e in events] == ["message_start"]


async def test_tolerates_crlf_frame_separators():
    raw = b'event: ping\r\ndata: {"type":"ping"}\r\n\r\n'
    events = [e async for e in iter_sse_events(feed(raw))]

    assert [e.name for e in events] == ["ping"]


async def test_multibyte_utf8_split_across_chunks_is_not_corrupted():
    whole = frame("content_block_delta", {"delta": {"text": "안녕하세요"}})
    # Split in the middle of a Hangul character's bytes.
    mid = len(whole) // 2
    events = [e async for e in iter_sse_events(feed(whole[:mid], whole[mid:]))]

    assert events[0].data["delta"]["text"] == "안녕하세요"


async def test_preserves_raw_bytes_for_reemission():
    original = frame("message_start", {"type": "message_start", "message": {"id": "msg_1"}})
    events = [e async for e in iter_sse_events(feed(original))]

    assert events[0].raw == original


async def test_data_only_frame_without_event_name():
    raw = b'data: {"type":"ping"}\n\n'
    events = [e async for e in iter_sse_events(feed(raw))]

    assert events[0].data["type"] == "ping"


async def test_openai_done_sentinel_is_surfaced():
    events = [e async for e in iter_sse_events(feed(b"data: [DONE]\n\n"))]

    assert events[0].is_done is True


async def test_trailing_frame_without_blank_line_is_still_emitted():
    events = [e async for e in iter_sse_events(feed(b'data: {"a":1}'))]
    assert events[0].data == {"a": 1}


# -- Anthropic re-emission ----------------------------------------------


def test_renders_a_valid_text_block_sequence():
    out = b"".join(render_anthropic_text_block(0, "Hello there"))
    text = out.decode()

    assert "event: content_block_start" in text
    assert "event: content_block_delta" in text
    assert "event: content_block_stop" in text
    assert text.index("content_block_start") < text.index("content_block_delta")
    assert text.index("content_block_delta") < text.index("content_block_stop")


def test_rendered_block_carries_the_translated_text_and_index():
    out = b"".join(render_anthropic_text_block(2, "안녕하세요")).decode()
    deltas = [
        json.loads(line[len("data: ") :])
        for line in out.splitlines()
        if line.startswith("data: ") and "text_delta" in line
    ]

    assert "".join(d["delta"]["text"] for d in deltas) == "안녕하세요"
    assert all(d["index"] == 2 for d in deltas)


def test_rendered_block_does_not_escape_non_ascii():
    out = b"".join(render_anthropic_text_block(0, "안녕")).decode()
    assert "안녕" in out
    assert "\\u" not in out


def test_long_text_is_split_into_multiple_deltas():
    long_text = "\n".join(f"line {i} " + "x" * 200 for i in range(60))
    out = b"".join(render_anthropic_text_block(0, long_text)).decode()
    delta_count = out.count("event: content_block_delta")

    assert delta_count > 1, "a large block should stream in several deltas"
    deltas = [
        json.loads(line[len("data: ") :])
        for line in out.splitlines()
        if line.startswith("data: ") and "text_delta" in line
    ]
    assert "".join(d["delta"]["text"] for d in deltas) == long_text


def test_empty_text_still_produces_a_well_formed_block():
    out = b"".join(render_anthropic_text_block(0, "")).decode()
    assert "content_block_start" in out and "content_block_stop" in out


# -- OpenAI re-emission -------------------------------------------------


def test_renders_openai_content_chunk_with_original_identity():
    template = {
        "id": "chatcmpl-1",
        "created": 123,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_1",
    }
    out = b"".join(render_openai_content_chunk(template, 0, "안녕하세요")).decode()
    payloads = [
        json.loads(line[len("data: ") :]) for line in out.splitlines() if line.startswith("data: ")
    ]

    assert all(p["id"] == "chatcmpl-1" for p in payloads)
    assert all(p["created"] == 123 for p in payloads)
    assert all(p["model"] == "gpt-4o-mini" for p in payloads)
    assert all(p["object"] == "chat.completion.chunk" for p in payloads)
    assert "".join(p["choices"][0]["delta"]["content"] for p in payloads) == "안녕하세요"


def test_openai_chunk_has_null_finish_reason():
    out = b"".join(render_openai_content_chunk({"id": "c"}, 0, "hi")).decode()
    payload = json.loads(
        [line for line in out.splitlines() if line.startswith("data: ")][0][len("data: ") :]
    )
    assert payload["choices"][0]["finish_reason"] is None


# -- ping ---------------------------------------------------------------


async def test_ping_is_emitted_while_a_translation_is_pending():
    """A silent stream is aborted by clients, so we must keep it alive."""
    from lingua_proxy.streaming import with_keepalive

    release = asyncio.Event()

    async def slow_work() -> list[bytes]:
        await release.wait()
        return [b"done\n\n"]

    seen: list[bytes] = []

    async def drain():
        async for chunk in with_keepalive(slow_work(), interval=0.02):
            seen.append(chunk)

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.11)
    pings = [c for c in seen if b"ping" in c]
    assert len(pings) >= 2, f"expected repeated pings, saw {len(pings)}"

    release.set()
    await task
    assert seen[-1] == b"done\n\n"


async def test_no_ping_when_work_completes_immediately():
    from lingua_proxy.streaming import with_keepalive

    async def fast_work() -> list[bytes]:
        return [b"done\n\n"]

    seen = [c async for c in with_keepalive(fast_work(), interval=5.0)]
    assert seen == [b"done\n\n"]


async def test_keepalive_propagates_the_underlying_error():
    from lingua_proxy.streaming import with_keepalive

    async def failing_work() -> list[bytes]:
        raise RuntimeError("translator died")

    with pytest.raises(RuntimeError):
        [c async for c in with_keepalive(failing_work(), interval=0.01)]


def test_sse_event_repr_is_debuggable():
    event = SSEEvent(name="ping", data={"type": "ping"}, raw=b"x")
    assert "ping" in repr(event)
