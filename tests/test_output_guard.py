"""Do not translate requests whose reply will be too long to be worth it.

Translation cost scales with the answer, while the saving does not. Past a
certain reply length the fee outruns the benefit, so the proxy declines rather
than quietly losing the user money.
"""

from __future__ import annotations

import httpx

from lingua_proxy.config import Settings
from lingua_proxy.proxy import create_app
from tests.conftest import Canned, FakeTranslator, RecordingTransport

UPSTREAM = "https://gw.example/anthropic/"
KOREAN = "이 함수가 왜 느린지 설명해 주고, 더 빠르게 만들 수 있는 방법을 알려줘."


def reply(text: str = "ok") -> dict:
    return {
        "id": "msg_1",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 20},
    }


def build(**kw):
    transport = RecordingTransport(Canned(json_body=reply()))
    settings = Settings(upstream_anthropic_url=UPSTREAM, memo_persist=False, **kw)
    translator = FakeTranslator()
    app = create_app(settings, transport=transport, translator=translator)
    return app, transport, translator


async def post(app, body, **kw):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as c:
        return await c.post("/v1/messages", json=body, **kw)


async def test_request_with_a_huge_max_tokens_is_not_translated():
    app, transport, translator = build(max_output_tokens_for_translation=4000)
    await post(
        app,
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32000,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert translator.call_count == 0, "translated a request whose reply will be long"
    assert transport.json_bodies[0]["messages"][0]["content"] == KOREAN


async def test_request_with_a_modest_max_tokens_is_translated():
    app, _, translator = build(max_output_tokens_for_translation=4000)
    await post(
        app,
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert translator.call_count > 0


async def test_the_guard_can_be_switched_off():
    app, _, translator = build(max_output_tokens_for_translation=0)
    await post(
        app,
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100000,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert translator.call_count > 0, "guard should be disabled at 0"


async def test_a_request_without_max_tokens_is_still_translated():
    """Absent a ceiling we cannot predict the length, so behave as before."""
    app, _, translator = build(max_output_tokens_for_translation=4000)
    await post(
        app, {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    )

    assert translator.call_count > 0


async def test_skipped_request_is_marked_as_untranslated():
    app, _, _ = build(max_output_tokens_for_translation=4000)
    resp = await post(
        app,
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32000,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert resp.headers.get("x-lingua-translated") == "false"


async def test_skip_reason_is_reported_so_a_user_can_tell_why():
    app, _, _ = build(max_output_tokens_for_translation=4000)
    resp = await post(
        app,
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32000,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    assert resp.headers.get("x-lingua-skipped") == "long_output"


async def test_bypass_header_reports_its_own_reason():
    app, _, _ = build()
    resp = await post(
        app,
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]},
        headers={"x-lingua-bypass": "true"},
    )

    assert resp.headers.get("x-lingua-skipped") == "bypass_header"
