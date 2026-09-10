"""The user must be able to see what the model was actually asked.

A wrong translation is fluent and structurally valid, so no automated check
catches it. The defence is not detection but visibility: whenever the proxy
rewrites a prompt, it says so, and the English it sent is recoverable.
"""

from __future__ import annotations

import json

import httpx

from lingua_proxy.config import Settings
from lingua_proxy.proxy import create_app
from tests.conftest import Canned, FakeTranslator, RecordingTransport

UPSTREAM = "https://gw.example/anthropic/"
KOREAN = "이 함수가 왜 느린지 설명해 주고, 더 빠르게 만들 수 있는 방법을 알려줘."


def reply(text: str) -> dict:
    return {
        "id": "msg_1",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 40},
    }


def build(**kw):
    transport = RecordingTransport(Canned(json_body=reply("The loop is O(n^2).")))
    settings = Settings(upstream_anthropic_url=UPSTREAM, memo_persist=False, **kw)
    app = create_app(settings, transport=transport, translator=FakeTranslator())
    return app, transport


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


async def post(app, body, **kw):
    async with client_for(app) as c:
        return await c.post("/v1/messages", json=body, **kw)


async def test_translated_response_announces_that_it_was_translated():
    app, _ = build()
    resp = await post(
        app, {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    )

    assert resp.headers.get("x-lingua-translated") == "true"
    assert resp.headers.get("x-lingua-source-lang") == "ko"


async def test_passthrough_response_is_marked_as_untranslated():
    app, _ = build()
    resp = await post(
        app,
        {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Explain why this function is slow."}],
        },
    )

    assert resp.headers.get("x-lingua-translated") == "false"


async def test_the_english_actually_sent_is_recoverable_from_the_response():
    """Without this, a user cannot tell a bad answer from a bad translation."""
    app, _ = build()
    resp = await post(
        app, {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    )

    from urllib.parse import unquote

    echoed = resp.headers.get("x-lingua-prompt-en")
    assert echoed, "the English prompt was not exposed"
    # The fake translator marks its output, so decoding proves this is the
    # translated text and not an echo of the original.
    assert "«en»" in unquote(echoed)


async def test_echo_header_is_percent_encoded_for_non_ascii_safety():
    """Header values must be latin-1 safe or the response cannot be sent."""
    app, _ = build()
    resp = await post(
        app, {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    )

    value = resp.headers.get("x-lingua-prompt-en", "")
    value.encode("latin-1")  # must not raise


async def test_review_mode_returns_the_english_without_calling_the_model():
    """Lets a user check a translation before spending a request on it."""
    app, transport = build()
    resp = await post(
        app,
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]},
        headers={"x-lingua-review": "true"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["lingua_review"] is True
    assert body["source_lang"] == "ko"
    assert body["translated_prompt"]
    assert transport.requests == [], "review mode must not reach the upstream model"


async def test_review_mode_reports_when_nothing_would_be_translated():
    app, transport = build()
    resp = await post(
        app,
        {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Explain why this function is slow."}],
        },
        headers={"x-lingua-review": "true"},
    )

    body = resp.json()
    assert body["lingua_review"] is True
    assert body["would_translate"] is False
    assert transport.requests == []


async def test_audit_log_records_both_sides_when_enabled(tmp_path):
    """Opt-in, because it writes prompt text to disk."""
    audit = tmp_path / "audit.jsonl"
    app, _ = build(audit_log_path=audit)
    await post(
        app, {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    )

    entries = [json.loads(ln) for ln in audit.read_text().splitlines() if ln.strip()]
    requests = [e for e in entries if e["direction"] == "request"]
    responses = [e for e in entries if e["direction"] == "response"]

    assert requests, "the rewritten prompt was not recorded"
    assert requests[0]["source_lang"] == "ko"
    assert requests[0]["original"] == KOREAN
    assert requests[0]["translated"] != KOREAN

    assert responses, "the translated reply was not recorded"
    assert responses[0]["target_lang"] == "ko"


async def test_audit_log_is_off_by_default(tmp_path):
    app, _ = build()
    await post(
        app, {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    )

    assert not list(tmp_path.glob("*.jsonl")), "audit log wrote without being enabled"


async def test_audit_log_permissions_are_owner_only(tmp_path):
    audit = tmp_path / "audit.jsonl"
    app, _ = build(audit_log_path=audit)
    await post(
        app, {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    )

    assert (audit.stat().st_mode & 0o077) == 0
