"""The test harness must be trustworthy before anything is built on it."""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest

from tests.conftest import Canned, FakeTranslator, RecordingTransport, sse


async def test_recording_transport_records_request_and_replays_script():
    transport = RecordingTransport(Canned(status=200, json_body={"ok": True}))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await client.post("https://upstream.test/v1/messages", json={"model": "m"})

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert len(transport.requests) == 1
    assert transport.requests[0].url.path == "/v1/messages"
    assert transport.json_bodies[0] == {"model": "m"}


async def test_recording_transport_consumes_script_in_order_then_repeats_last():
    transport = RecordingTransport([Canned(json_body={"n": 1}), Canned(json_body={"n": 2})])
    async with httpx.AsyncClient(transport=transport) as client:
        first = await client.get("https://upstream.test/a")
        second = await client.get("https://upstream.test/b")
        third = await client.get("https://upstream.test/c")

    assert [first.json(), second.json(), third.json()] == [{"n": 1}, {"n": 2}, {"n": 2}]


async def test_recording_transport_streams_chunks_separately():
    chunks = sse(
        ("message_start", {"type": "message_start"}),
        ("message_stop", {"type": "message_stop"}),
    )
    transport = RecordingTransport(Canned(chunks=chunks))
    seen: list[bytes] = []
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("POST", "https://upstream.test/v1/messages") as resp:
            assert resp.headers["content-type"] == "text/event-stream"
            async for chunk in resp.aiter_raw():
                seen.append(chunk)

    assert seen == chunks
    assert b"message_start" in seen[0]


async def test_chunk_gate_holds_a_chunk_until_released():
    """Proves we can assert ordering: chunk 2 must not arrive before release."""
    gate = asyncio.Event()
    chunks = [b"first\n\n", b"second\n\n"]
    transport = RecordingTransport(Canned(chunks=chunks, chunk_gates=[None, gate]))
    seen: list[bytes] = []

    async def read() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            async with client.stream("GET", "https://upstream.test/s") as resp:
                async for chunk in resp.aiter_raw():
                    seen.append(chunk)

    task = asyncio.create_task(read())
    await asyncio.sleep(0.05)
    assert seen == [b"first\n\n"], "second chunk leaked before its gate was released"
    gate.set()
    await task
    assert seen == chunks


async def test_recording_transport_raises_configured_exception():
    transport = RecordingTransport(Canned(exc=httpx.ConnectError("refused")))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("https://upstream.test/x")


async def test_fake_translator_is_deterministic_and_records_calls():
    t = FakeTranslator()
    out = await t.translate(["안녕", "반가워"], "ko", "en")

    assert out == ["«en»안녕", "«en»반가워"]
    assert await t.translate(["안녕"], "ko", "en") == ["«en»안녕"]
    assert t.call_count == 2
    assert t.calls[0] == (["안녕", "반가워"], "ko", "en")


async def test_fake_translator_preserves_placeholders():
    t = FakeTranslator()
    (out,) = await t.translate(["코드 <lp0/> 를 봐"], "ko", "en")
    assert "<lp0/>" in out


async def test_fake_translator_honours_mapping_and_failure():
    t = FakeTranslator({"안녕": "hello"})
    assert await t.translate(["안녕"], "ko", "en") == ["hello"]

    t.fail_with = RuntimeError("model down")
    with pytest.raises(RuntimeError):
        await t.translate(["안녕"], "ko", "en")


def test_isolated_home_redirects_home_away_from_real_user(isolated_home):
    assert os.environ["HOME"] == str(isolated_home)
    assert "Desktop" not in os.environ["HOME"]
    assert os.environ["LINGUA_HOME"].endswith(".lingua-proxy")


def test_isolated_home_clears_upstream_env_vars():
    assert "ANTHROPIC_BASE_URL" not in os.environ
    assert "LINGUA_UPSTREAM_ANTHROPIC_URL" not in os.environ


def test_sse_builds_wire_format_frames():
    (frame,) = sse(("ping", {"type": "ping"}))
    assert frame == b'event: ping\ndata: {"type":"ping"}\n\n'


def test_sse_does_not_escape_non_ascii():
    (frame,) = sse(("content_block_delta", {"text": "안녕"}))
    assert "안녕" in frame.decode()
    assert json.loads(frame.decode().split("data: ", 1)[1])["text"] == "안녕"
