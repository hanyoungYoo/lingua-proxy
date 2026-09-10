"""Step 1: the proxy must be a faithful reverse proxy before it translates anything.

Everything here is about fidelity: bytes, headers, status codes and streaming
timing must survive the hop unchanged.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from lingua_proxy import __version__
from lingua_proxy.config import Settings
from lingua_proxy.proxy import create_app
from tests.conftest import Canned, RecordingTransport, sse

UPSTREAM = "https://gw.example/anthropic/"


def build(script, **kw) -> tuple[object, RecordingTransport]:
    transport = RecordingTransport(script)
    settings = Settings(upstream_anthropic_url=UPSTREAM, **kw)
    app = create_app(settings, transport=transport)
    return app, transport


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


async def test_healthz_identifies_the_service():
    app, _ = build(Canned(json_body={}))
    async with client_for(app) as c:
        resp = await c.get("/healthz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "lingua-proxy"
    assert body["version"] == __version__


async def test_request_body_is_forwarded_byte_exact():
    app, transport = build(Canned(json_body={"ok": True}))
    payload = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]}
    raw = json.dumps(payload).encode()

    async with client_for(app) as c:
        await c.post("/v1/messages", content=raw, headers={"content-type": "application/json"})

    assert transport.bodies[0] == raw


async def test_upstream_path_prefix_and_query_are_preserved():
    app, transport = build(Canned(json_body={}))
    async with client_for(app) as c:
        await c.post("/v1/messages?beta=true", json={"model": "m"})

    url = transport.requests[0].url
    assert str(url) == "https://gw.example/anthropic/v1/messages?beta=true"


async def test_client_headers_including_anthropic_and_claude_code_are_forwarded():
    app, transport = build(Canned(json_body={}))
    headers = {
        "authorization": "Bearer secret-token",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "oauth-2025-04-20,context-management-2025-06-27",
        "x-claude-code-session-id": "sess-123",
        "user-agent": "claude-cli/2.1.0",
    }
    async with client_for(app) as c:
        await c.post("/v1/messages", json={"model": "m"}, headers=headers)

    sent = transport.requests[0].headers
    for key, value in headers.items():
        assert sent[key] == value, f"{key} was not forwarded verbatim"


async def test_x_api_key_credential_style_is_forwarded():
    app, transport = build(Canned(json_body={}))
    async with client_for(app) as c:
        await c.post("/v1/messages", json={"model": "m"}, headers={"x-api-key": "sk-test"})

    assert transport.requests[0].headers["x-api-key"] == "sk-test"


async def test_accept_encoding_identity_is_sent_so_relay_can_be_byte_exact():
    app, transport = build(Canned(json_body={}))
    async with client_for(app) as c:
        await c.post("/v1/messages", json={"model": "m"}, headers={"accept-encoding": "gzip, br"})

    assert transport.requests[0].headers["accept-encoding"] == "identity"


async def test_host_and_internal_headers_are_not_forwarded():
    app, transport = build(Canned(json_body={}))
    async with client_for(app) as c:
        await c.post(
            "/v1/messages",
            json={"model": "m"},
            headers={"x-lingua-bypass": "true"},
        )

    sent = transport.requests[0].headers
    assert "x-lingua-bypass" not in sent
    assert sent.get("host") != "proxy.test"


async def test_response_body_and_status_are_relayed():
    upstream_body = {"id": "msg_1", "content": [{"type": "text", "text": "hello"}]}
    app, _ = build(Canned(status=200, json_body=upstream_body))
    async with client_for(app) as c:
        resp = await c.post("/v1/messages", json={"model": "m"})

    assert resp.status_code == 200
    assert resp.json() == upstream_body


@pytest.mark.parametrize("status", [400, 401, 429, 500, 529])
async def test_upstream_error_bodies_are_relayed_unmodified(status):
    """Clients match on the upstream's own error wording; we must not reshape it."""
    err = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    app, _ = build(Canned(status=status, json_body=err))
    async with client_for(app) as c:
        resp = await c.post("/v1/messages", json={"model": "m"})

    assert resp.status_code == status
    assert resp.json() == err


async def test_upstream_connect_failure_becomes_502_with_typed_error():
    app, _ = build(Canned(exc=httpx.ConnectError("connection refused")))
    async with client_for(app) as c:
        resp = await c.post("/v1/messages", json={"model": "m"})

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"]["type"] == "lingua_proxy_upstream_error"


async def test_catch_all_passes_through_count_tokens():
    app, transport = build(Canned(json_body={"input_tokens": 42}))
    async with client_for(app) as c:
        resp = await c.post("/v1/messages/count_tokens", json={"model": "m", "messages": []})

    assert resp.json() == {"input_tokens": 42}
    assert transport.requests[0].url.path == "/anthropic/v1/messages/count_tokens"


async def test_catch_all_passes_through_head_api_hello():
    app, transport = build(Canned(status=200))
    async with client_for(app) as c:
        resp = await c.head("/api/hello")

    assert resp.status_code == 200
    assert transport.requests[0].method == "HEAD"


async def test_catch_all_passes_through_get_models():
    app, transport = build(Canned(json_body={"data": []}))
    async with client_for(app) as c:
        resp = await c.get("/v1/models?limit=1000")

    assert resp.status_code == 200
    assert str(transport.requests[0].url).endswith("/anthropic/v1/models?limit=1000")


async def test_streaming_response_frames_are_relayed_unchanged():
    chunks = sse(
        ("message_start", {"type": "message_start", "message": {"id": "msg_1"}}),
        ("content_block_delta", {"type": "content_block_delta", "delta": {"text": "안녕"}}),
        ("message_stop", {"type": "message_stop"}),
    )
    app, _ = build(Canned(chunks=chunks))

    async with client_for(app) as c:
        async with c.stream("POST", "/v1/messages", json={"model": "m", "stream": True}) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            received = b"".join([chunk async for chunk in resp.aiter_raw()])

    assert received == b"".join(chunks)


async def test_hop_by_hop_headers_are_stripped_from_response():
    app, _ = build(
        Canned(
            json_body={},
            headers={"transfer-encoding": "chunked", "connection": "keep-alive", "x-req-id": "r1"},
        )
    )
    async with client_for(app) as c:
        resp = await c.post("/v1/messages", json={"model": "m"})

    assert "transfer-encoding" not in resp.headers
    assert "connection" not in resp.headers
    assert resp.headers["x-req-id"] == "r1"


async def test_openai_route_uses_the_openai_upstream():
    transport = RecordingTransport(Canned(json_body={"ok": True}))
    settings = Settings(
        upstream_anthropic_url=UPSTREAM,
        upstream_openai_url="https://gw.example/openai/",
    )
    app = create_app(settings, transport=transport)

    async with client_for(app) as c:
        await c.post("/v1/chat/completions", json={"model": "gpt-4o-mini"})

    assert str(transport.requests[0].url) == "https://gw.example/openai/v1/chat/completions"


async def test_relay_does_not_buffer_the_upstream_stream():
    """The first frame must reach the client before the last one is produced.

    ASGITransport buffers whole responses, so this exercises the relay
    generator directly with a gated upstream.
    """
    from lingua_proxy.proxy import stream_upstream

    gate = asyncio.Event()
    chunks = [b"first\n\n", b"second\n\n"]
    transport = RecordingTransport(Canned(chunks=chunks, chunk_gates=[None, gate]))
    seen: list[bytes] = []

    async with httpx.AsyncClient(transport=transport) as upstream_client:
        request = upstream_client.build_request("GET", "https://gw.example/stream")
        response = await upstream_client.send(request, stream=True)

        async def drain():
            async for chunk in stream_upstream(response):
                seen.append(chunk)

        task = asyncio.create_task(drain())
        await asyncio.sleep(0.05)
        assert seen == [b"first\n\n"], "relay buffered instead of streaming"
        gate.set()
        await task

    assert seen == chunks
