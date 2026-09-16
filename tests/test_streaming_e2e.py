"""Streaming through the whole proxy, including with the official SDKs.

The SDK tests are the strongest offline evidence that a real client will
accept the events we synthesize.
"""

from __future__ import annotations

import json

import httpx

from lingua_proxy.config import Settings
from lingua_proxy.proxy import create_app
from tests.conftest import Canned, FakeTranslator, RecordingTransport

UPSTREAM = "https://gw.example/anthropic/"
KOREAN = "이 함수가 왜 느린지 설명해 주고, 더 빠르게 만들 수 있는 방법을 알려줘."


def frame(name: str, data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n".encode()


def anthropic_stream(*texts: str, with_tool: bool = False) -> list[bytes]:
    chunks = [
        frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                },
            },
        )
    ]
    index = 0
    for text in texts:
        chunks.append(
            frame(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        for piece in text.split(" "):
            chunks.append(
                frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": piece + " "},
                    },
                )
            )
        chunks.append(frame("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1

    if with_tool:
        chunks.append(
            frame(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "bash",
                        "input": {},
                    },
                },
            )
        )
        chunks.append(
            frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": '{"cmd":"ls"}'},
                },
            )
        )
        chunks.append(frame("content_block_stop", {"type": "content_block_stop", "index": index}))

    chunks.append(
        frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 42},
            },
        )
    )
    chunks.append(frame("message_stop", {"type": "message_stop"}))
    return chunks


def build(chunks, translator=None, **kw):
    transport = RecordingTransport(Canned(chunks=chunks))
    settings = Settings(
        upstream_anthropic_url=UPSTREAM, memo_persist=False, ping_interval=0.05, **kw
    )
    translator = translator or FakeTranslator()
    app = create_app(settings, transport=transport, translator=translator)
    return app, transport, translator


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


async def collect(app, body) -> str:
    async with client_for(app) as c:
        async with c.stream("POST", "/v1/messages", json=body) as resp:
            return (await resp.aread()).decode()


async def test_streamed_text_is_translated():
    app, _, _ = build(anthropic_stream("The function is slow."))
    out = await collect(
        app,
        {
            "model": "claude-sonnet-4-6",
            "stream": True,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert "«ko»" in out, "streamed text was not translated"
    assert "event: content_block_start" in out
    assert "event: message_stop" in out


async def test_a_streamed_reply_says_what_the_model_was_actually_asked():
    """Transparency cannot only work on the path clients do not use.

    Agentic clients stream by default, so a debugging story that relies on
    x-lingua-prompt-en is worth nothing if the header is absent on exactly
    the traffic people run.
    """
    app, _, _ = build(anthropic_stream("The function is slow."))
    async with client_for(app) as c:
        async with c.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "stream": True,
                "messages": [{"role": "user", "content": KOREAN}],
            },
        ) as resp:
            await resp.aread()
            assert resp.headers.get("x-lingua-translated") == "true"
            assert resp.headers.get("x-lingua-source-lang") == "ko"
            assert resp.headers.get("x-lingua-prompt-en"), (
                "the English actually sent is not recoverable from a streamed reply"
            )


async def test_an_untranslated_stream_says_so_too():
    app, _, _ = build(anthropic_stream("Already English."))
    async with client_for(app) as c:
        async with c.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "stream": True,
                "messages": [{"role": "user", "content": "Why is this function slow?"}],
            },
        ) as resp:
            await resp.aread()
            assert resp.headers.get("x-lingua-translated") == "false"


async def test_message_start_and_usage_are_preserved():
    app, _, _ = build(anthropic_stream("Hello."))
    out = await collect(
        app,
        {
            "model": "claude-sonnet-4-6",
            "stream": True,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert '"input_tokens":100' in out
    assert '"output_tokens":42' in out
    assert '"id":"msg_1"' in out


async def test_tool_use_blocks_pass_through_byte_identical():
    """Tool call arguments must never be reserialized or translated."""
    app, _, _ = build(anthropic_stream("Running it.", with_tool=True))
    out = await collect(
        app,
        {
            "model": "claude-sonnet-4-6",
            "stream": True,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert '"partial_json":"{\\"cmd\\":\\"ls\\"}"' in out
    assert '"name":"bash"' in out


async def test_multiple_text_blocks_are_each_translated():
    app, _, translator = build(anthropic_stream("First part.", "Second part."))
    out = await collect(
        app,
        {
            "model": "claude-sonnet-4-6",
            "stream": True,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert out.count("event: content_block_stop") == 2
    assert "First part." in " ".join(translator.translated_segments)
    assert "Second part." in " ".join(translator.translated_segments)


async def test_english_stream_is_relayed_untouched():
    chunks = anthropic_stream("Hello there.")
    app, _, translator = build(chunks)
    out = await collect(
        app,
        {
            "model": "claude-sonnet-4-6",
            "stream": True,
            "messages": [{"role": "user", "content": "Explain why this function is slow."}],
        },
    )

    assert translator.call_count == 0
    assert out == b"".join(chunks).decode()


async def test_bypass_header_relays_the_stream_untouched():
    chunks = anthropic_stream("Hello there.")
    transport = RecordingTransport(Canned(chunks=chunks))
    settings = Settings(upstream_anthropic_url=UPSTREAM, memo_persist=False)
    translator = FakeTranslator()
    app = create_app(settings, transport=transport, translator=translator)

    async with client_for(app) as c:
        async with c.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "stream": True,
                "messages": [{"role": "user", "content": KOREAN}],
            },
            headers={"x-lingua-bypass": "true"},
        ) as resp:
            out = (await resp.aread()).decode()

    assert translator.call_count == 0
    assert out == b"".join(chunks).decode()


async def test_translator_failure_streams_the_english_answer():
    translator = FakeTranslator()
    app, _, _ = build(anthropic_stream("The function is slow."), translator)
    translator.fail_with = RuntimeError("model down")

    out = await collect(
        app,
        {
            "model": "claude-sonnet-4-6",
            "stream": True,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    # Passthrough preserves the upstream's own delta boundaries, so the words
    # arrive split exactly as the model emitted them.
    import re

    deltas = re.findall(r'"text_delta","text":"([^"]*)"', out)
    assert "".join(deltas) == "The function is slow. ", "the English answer was lost"
    assert "«ko»" not in out
    assert "event: message_stop" in out


async def test_truncated_upstream_stream_yields_an_error_event():
    partial = anthropic_stream("Hello.")[:3]  # cut before message_stop
    app, _, _ = build(partial)

    out = await collect(
        app,
        {
            "model": "claude-sonnet-4-6",
            "stream": True,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert "event: error" in out
    assert "before message_stop" in out


async def test_upstream_error_status_is_relayed_for_streaming_requests():
    transport = RecordingTransport(
        Canned(status=429, json_body={"type": "error", "error": {"type": "rate_limit_error"}})
    )
    settings = Settings(upstream_anthropic_url=UPSTREAM, memo_persist=False)
    app = create_app(settings, transport=transport, translator=FakeTranslator())

    async with client_for(app) as c:
        resp = await c.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "stream": True,
                "messages": [{"role": "user", "content": KOREAN}],
            },
        )

    assert resp.status_code == 429
    assert resp.json()["error"]["type"] == "rate_limit_error"


async def test_official_anthropic_sdk_parses_the_translated_stream():
    """The strongest offline proof that a real client accepts our events."""
    import httpx2
    from anthropic import AsyncAnthropic

    app, _, _ = build(anthropic_stream("The function is slow."))
    sdk = AsyncAnthropic(
        api_key="test-key",
        base_url="http://proxy.test",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
    )

    async with sdk.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": KOREAN}],
    ) as stream:
        message = await stream.get_final_message()

    assert message.content[0].text.startswith("«ko»")
    assert message.usage.input_tokens == 100


async def test_official_anthropic_sdk_sees_tool_use_intact():
    import httpx2
    from anthropic import AsyncAnthropic

    app, _, _ = build(anthropic_stream("Running it.", with_tool=True))
    sdk = AsyncAnthropic(
        api_key="test-key",
        base_url="http://proxy.test",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
    )

    async with sdk.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": KOREAN}],
    ) as stream:
        message = await stream.get_final_message()

    tool_blocks = [b for b in message.content if b.type == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "bash"
    assert tool_blocks[0].input == {"cmd": "ls"}
