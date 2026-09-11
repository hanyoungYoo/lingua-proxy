"""Keep the answer's shape, not just its facts.

Models format differently by language: the same question answered in Korean
comes back with a heading and a numbered list, answered in English it comes
back as prose. Translating that English gives a faithful translation of a
differently-shaped answer, so the user silently loses structure they never
agreed to trade away.

The fix is an instruction to the model, not to the translator -- the
translator was already preserving markdown correctly.
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
    app = create_app(settings, transport=transport, translator=FakeTranslator())
    return app, transport


async def post(app, body, **kw):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as c:
        return await c.post("/v1/messages", json=body, **kw)


async def test_style_hint_is_on_by_default():
    """Losing formatting was never something users opted into."""
    app, transport = build()
    await post(
        app, {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    )

    system = str(transport.json_bodies[0].get("system", "")).lower()
    assert "headings" in system and "lists" in system


async def test_style_hint_can_be_switched_off():
    app, transport = build(preserve_formatting=False)
    await post(
        app, {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    )

    assert "system" not in transport.json_bodies[0]


async def test_hint_is_appended_to_an_existing_string_system_prompt():
    app, transport = build()
    await post(
        app,
        {
            "model": "claude-sonnet-4-6",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    system = transport.json_bodies[0]["system"]
    assert isinstance(system, list), "a string system prompt should become a block list"
    assert system[0]["text"] == "You are a helpful assistant."
    assert "headings" in system[1]["text"].lower()


async def test_hint_is_appended_to_a_block_system_prompt_without_disturbing_it():
    """Block form carries cache_control; the existing blocks must be untouched."""
    original = [{"type": "text", "text": "Base prompt.", "cache_control": {"type": "ephemeral"}}]
    app, transport = build()
    await post(
        app,
        {
            "model": "claude-sonnet-4-6",
            "system": original,
            "messages": [{"role": "user", "content": KOREAN}],
        },
    )

    system = transport.json_bodies[0]["system"]
    assert system[0] == original[0], "the cached system block was modified"
    assert len(system) == 2
    assert "headings" in system[1]["text"].lower()


async def test_no_hint_when_the_request_is_not_translated():
    """An English request must reach the upstream completely untouched."""
    app, transport = build()
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Explain why this function is slow."}],
    }
    await post(app, body)

    assert transport.json_bodies[0] == body


async def test_no_hint_when_bypassed():
    app, transport = build()
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    await post(app, body, headers={"x-lingua-bypass": "true"})

    assert transport.json_bodies[0] == body


async def test_hint_can_be_disabled_per_request():
    """Some callers control formatting themselves and want nothing injected."""
    app, transport = build()
    await post(
        app,
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]},
        headers={"x-lingua-preserve-formatting": "false"},
    )

    assert "system" not in transport.json_bodies[0]


async def test_instruction_asks_for_structure_matching_the_question():
    """A question asking for three points should come back as three items.

    The first version of this instruction said only "keep your usual
    formatting", which produced bullets where the native answer used a
    numbered list. Naming the mirroring rule explicitly fixed 2 of 3 sampled
    prompts against 1 of 3.
    """
    from lingua_proxy.proxy import STYLE_INSTRUCTION

    lowered = STYLE_INSTRUCTION.lower()
    assert "numbered" in lowered
    assert "original language" in lowered


async def test_instruction_still_forbids_flattening_and_shortening():
    from lingua_proxy.proxy import STYLE_INSTRUCTION

    lowered = STYLE_INSTRUCTION.lower()
    assert "not flatten" in lowered
    assert "not shorten" in lowered
