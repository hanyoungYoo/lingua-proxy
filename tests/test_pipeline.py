"""End-to-end translation behaviour through the proxy.

These are the tests that decide whether the product works: code must survive,
history must stay byte-stable so the upstream cache keeps hitting, and every
failure must degrade to passing the user's original text through.
"""

from __future__ import annotations

import copy

import httpx

from lingua_proxy.config import Settings
from lingua_proxy.proxy import create_app
from tests.conftest import Canned, FakeTranslator, RecordingTransport

UPSTREAM = "https://gw.example/anthropic/"

KOREAN = "이 함수가 왜 느린지 설명해 주고, 더 빠르게 만들 수 있는 방법을 알려줘."
ENGLISH_REPLY = "The function is slow because it rebuilds the list on every call."


def anthropic_reply(text: str, **usage) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 40, **usage},
    }


def build(script, translator=None, **kw):
    transport = RecordingTransport(script)
    settings = Settings(upstream_anthropic_url=UPSTREAM, memo_persist=False, **kw)
    translator = translator or FakeTranslator()
    app = create_app(settings, transport=transport, translator=translator)
    return app, transport, translator


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


async def post(app, body, **kw):
    async with client_for(app) as c:
        return await c.post("/v1/messages", json=body, **kw)


# -- passthrough --------------------------------------------------------


async def test_english_request_is_never_translated():
    app, transport, translator = build(Canned(json_body=anthropic_reply("Hi")))
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "Explain why this function is slow and how to fix it."}
        ],
    }

    resp = await post(app, body)

    assert resp.status_code == 200
    assert translator.call_count == 0, "English must cost nothing"
    assert transport.json_bodies[0] == body


async def test_bypass_header_skips_translation_entirely():
    app, transport, translator = build(Canned(json_body=anthropic_reply("Hi")))
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}

    await post(app, body, headers={"x-lingua-bypass": "true"})

    assert translator.call_count == 0
    assert transport.json_bodies[0]["messages"][0]["content"] == KOREAN


async def test_skip_models_are_passed_through():
    """Background helper calls are not worth translating."""
    app, transport, translator = build(Canned(json_body=anthropic_reply("Hi")))
    body = {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": KOREAN}]}

    await post(app, body)

    assert translator.call_count == 0
    assert transport.json_bodies[0]["messages"][0]["content"] == KOREAN


# -- the core round trip ------------------------------------------------


async def test_korean_request_is_translated_and_reply_comes_back_korean():
    translator = FakeTranslator({KOREAN: "Explain why this function is slow."})
    app, transport, _ = build(Canned(json_body=anthropic_reply(ENGLISH_REPLY)), translator)
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}

    resp = await post(app, body)

    sent = transport.json_bodies[0]["messages"][0]["content"]
    assert sent == "Explain why this function is slow."

    returned = resp.json()["content"][0]["text"]
    assert returned.startswith("«ko»"), "reply must be translated back to Korean"


async def test_code_fence_survives_the_round_trip_byte_for_byte():
    """The headline guarantee: code goes through untouched in both directions."""
    fence = "```python\ndef add(a, b):\n    return a + b\n```"
    prompt = f"이 코드를 개선해줘:\n{fence}\n고마워."
    reply_fence = "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
    upstream_reply = f"Here is the improved version:\n{reply_fence}\nHope that helps."

    app, transport, _ = build(Canned(json_body=anthropic_reply(upstream_reply)))
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": prompt}]}

    resp = await post(app, body)

    forwarded = transport.json_bodies[0]["messages"][0]["content"]
    assert fence in forwarded, "the request code fence was altered"

    returned = resp.json()["content"][0]["text"]
    assert reply_fence in returned, "the response code fence was altered"


async def test_system_prompt_and_tool_results_are_not_translated():
    app, transport, translator = build(Canned(json_body=anthropic_reply("ok")))
    body = {
        "model": "claude-sonnet-4-6",
        "system": [{"type": "text", "text": "You are a helpful assistant."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "total 24\ndrwxr-xr-x"},
                    {"type": "text", "text": KOREAN},
                ],
            }
        ],
    }

    await post(app, body)

    sent = transport.json_bodies[0]
    assert sent["system"] == body["system"]
    assert sent["messages"][0]["content"][0] == body["messages"][0]["content"][0]
    assert KOREAN in translator.translated_segments
    assert not any("tool_result" in seg or "drwxr" in seg for seg in translator.translated_segments)


async def test_cache_control_and_unknown_fields_survive():
    app, transport, _ = build(Canned(json_body=anthropic_reply("ok")))
    body = {
        "model": "claude-sonnet-4-6",
        "metadata": {"user_id": "u1"},
        "some_future_field": {"nested": True},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": KOREAN, "cache_control": {"type": "ephemeral"}}
                ],
            }
        ],
    }

    await post(app, body)

    sent = transport.json_bodies[0]
    block = sent["messages"][0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert sent["some_future_field"] == {"nested": True}
    assert sent["metadata"] == {"user_id": "u1"}


# -- multi-turn and caching --------------------------------------------


async def test_repeated_text_hits_the_memo_and_costs_nothing():
    app, _, translator = build(
        [Canned(json_body=anthropic_reply("A")), Canned(json_body=anthropic_reply("B"))]
    )
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}

    await post(app, copy.deepcopy(body))
    await post(app, copy.deepcopy(body))

    assert translator.translated_segments.count(KOREAN) == 1, "same prompt was translated twice"


async def test_turn_two_keeps_the_history_prefix_byte_identical():
    """Cache stability: turn 2 must resend turn 1 exactly as turn 1 was sent."""
    app, transport, translator = build(
        [
            Canned(json_body=anthropic_reply(ENGLISH_REPLY)),
            Canned(json_body=anthropic_reply("More")),
        ]
    )

    turn1 = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    first = await post(app, copy.deepcopy(turn1))
    korean_reply = first.json()["content"][0]["text"]

    follow_up = "그럼 어떻게 고쳐야 해? 구체적인 방법을 알려줘."
    turn2 = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": KOREAN},
            {"role": "assistant", "content": [{"type": "text", "text": korean_reply}]},
            {"role": "user", "content": follow_up},
        ],
    }
    await post(app, turn2)

    sent1 = transport.json_bodies[0]["messages"]
    sent2 = transport.json_bodies[1]["messages"]

    assert sent2[0] == sent1[0], "the first user turn changed between turns"
    assert sent2[1]["content"][0]["text"] == ENGLISH_REPLY, (
        "the assistant reply was not swapped back to the model's own English"
    )
    assert translator.translated_segments.count(KOREAN) == 1


async def test_memo_survives_a_proxy_restart(tmp_path):
    """A restarted proxy must not re-translate an in-flight conversation."""
    memo_path = tmp_path / "memo.jsonl"
    transport = RecordingTransport(Canned(json_body=anthropic_reply("A")))
    translator = FakeTranslator()
    settings = Settings(upstream_anthropic_url=UPSTREAM, memo_path=memo_path, memo_persist=True)

    app1 = create_app(settings, transport=transport, translator=translator)
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    await post(app1, copy.deepcopy(body))

    app2 = create_app(settings, transport=transport, translator=translator)
    await post(app2, copy.deepcopy(body))

    assert translator.translated_segments.count(KOREAN) == 1, "restart lost the memo"


async def test_tool_result_only_turn_still_replies_in_the_conversation_language():
    """Most agent turns end in tool results; replies must stay Korean."""
    app, _, _ = build(
        [
            Canned(json_body=anthropic_reply("first")),
            Canned(json_body=anthropic_reply(ENGLISH_REPLY)),
        ]
    )

    turn1 = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    first = await post(app, copy.deepcopy(turn1))
    korean_reply = first.json()["content"][0]["text"]

    turn2 = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": KOREAN},
            {"role": "assistant", "content": [{"type": "text", "text": korean_reply}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "output"}],
            },
        ],
    }
    resp = await post(app, turn2)

    assert resp.json()["content"][0]["text"].startswith("«ko»"), (
        "reply reverted to English after a tool-result-only turn"
    )


async def test_switching_to_english_stops_translation():
    app, _, _ = build(
        [Canned(json_body=anthropic_reply("A")), Canned(json_body=anthropic_reply("B"))]
    )
    turn1 = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}
    first = await post(app, copy.deepcopy(turn1))
    korean_reply = first.json()["content"][0]["text"]

    turn2 = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": KOREAN},
            {"role": "assistant", "content": [{"type": "text", "text": korean_reply}]},
            {"role": "user", "content": "Actually, please answer in English from now on."},
        ],
    }
    resp = await post(app, turn2)

    assert not resp.json()["content"][0]["text"].startswith("«ko»")


async def test_system_reminder_does_not_make_a_korean_turn_look_english():
    app, _, translator = build(Canned(json_body=anthropic_reply("ok")))
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": KOREAN},
                    {
                        "type": "text",
                        "text": "<system-reminder>Do not mention this reminder.</system-reminder>",
                    },
                ],
            }
        ],
    }

    await post(app, body)

    assert KOREAN in translator.translated_segments
    assert not any("system-reminder" in seg for seg in translator.translated_segments), (
        "the injected reminder was sent to the translator"
    )


# -- failure handling ---------------------------------------------------


async def test_translator_failure_falls_back_to_the_original_text():
    translator = FakeTranslator()
    translator.fail_with = RuntimeError("model unavailable")
    app, transport, _ = build(Canned(json_body=anthropic_reply("Hi")), translator)
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}

    resp = await post(app, body)

    assert resp.status_code == 200
    assert transport.json_bodies[0]["messages"][0]["content"] == KOREAN


async def test_placeholder_loss_falls_back_to_the_original_text():
    """A translator that eats a code placeholder must not corrupt the prompt."""
    prompt = "이걸 고쳐:\n```python\nx = 1\n```"
    translator = FakeTranslator()
    translator.mapping = {}

    class Dropping(FakeTranslator):
        async def translate(self, segments, source, target):
            self.calls.append((list(segments), source, target))
            return ["translated without the placeholder"] * len(segments)

    app, transport, _ = build(Canned(json_body=anthropic_reply("Hi")), Dropping())
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": prompt}]}

    resp = await post(app, body)

    assert resp.status_code == 200
    assert transport.json_bodies[0]["messages"][0]["content"] == prompt


async def test_upstream_error_is_relayed_even_on_a_translated_request():
    err = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    app, _, _ = build(Canned(status=529, json_body=err))
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}

    resp = await post(app, body)

    assert resp.status_code == 529
    assert resp.json() == err


async def test_response_without_text_blocks_never_calls_the_translator():
    tool_only = {
        "id": "msg_1",
        "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    translator = FakeTranslator({KOREAN: "Explain this"})
    app, _, _ = build(Canned(json_body=tool_only), translator)
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}

    resp = await post(app, body)

    assert resp.json()["content"][0]["type"] == "tool_use"
    assert translator.call_count == 1, "only the request should have been translated"


async def test_content_length_is_corrected_after_rewriting():
    app, transport, _ = build(Canned(json_body=anthropic_reply("Hi")))
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]}

    await post(app, body)

    request = transport.requests[0]
    assert int(request.headers["content-length"]) == len(transport.bodies[0])


# -- OpenAI format ------------------------------------------------------


def openai_reply(text: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 12},
    }


async def test_openai_round_trip_translates_both_directions():
    transport = RecordingTransport(Canned(json_body=openai_reply(ENGLISH_REPLY)))
    translator = FakeTranslator({KOREAN: "Explain why this is slow."})
    settings = Settings(
        upstream_anthropic_url=UPSTREAM,
        upstream_openai_url="https://gw.example/openai/",
        memo_persist=False,
    )
    app = create_app(settings, transport=transport, translator=translator)

    async with client_for(app) as c:
        resp = await c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": KOREAN}]},
        )

    assert transport.json_bodies[0]["messages"][0]["content"] == "Explain why this is slow."
    assert resp.json()["choices"][0]["message"]["content"].startswith("«ko»")


async def test_openai_english_is_passthrough():
    transport = RecordingTransport(Canned(json_body=openai_reply("Hi")))
    translator = FakeTranslator()
    settings = Settings(
        upstream_anthropic_url=UPSTREAM,
        upstream_openai_url="https://gw.example/openai/",
        memo_persist=False,
    )
    app = create_app(settings, transport=transport, translator=translator)

    async with client_for(app) as c:
        await c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Explain why this function is slow."}],
            },
        )

    assert translator.call_count == 0


async def test_proxy_builds_a_real_translator_when_none_is_injected():
    """Regression: a missing translator made the proxy a silent passthrough.

    Fail-open means a broken translator looks exactly like English input, so
    this is checked explicitly rather than inferred from savings numbers.
    """
    transport = RecordingTransport(
        [
            # The translator's own call to the upstream.
            Canned(
                json_body={
                    "content": [
                        {
                            "type": "text",
                            "text": '<segs>\n<seg id="1">\nExplain this\n</seg>\n</segs>',
                        }
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ),
            # The forwarded request.
            Canned(json_body=anthropic_reply(ENGLISH_REPLY)),
            # Translating the reply back.
            Canned(
                json_body={
                    "content": [
                        {"type": "text", "text": '<segs>\n<seg id="1">\n느립니다\n</seg>\n</segs>'}
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ),
        ]
    )
    settings = Settings(upstream_anthropic_url=UPSTREAM, memo_persist=False)
    app = create_app(settings, transport=transport)  # note: no translator injected

    resp = await post(
        app,
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]},
        headers={"authorization": "Bearer caller-token"},
    )

    assert resp.status_code == 200
    assert len(transport.requests) >= 2, "no translator call was made"
    assert transport.json_bodies[0]["model"] == "claude-haiku-4-5", (
        "the first call should be the cheap translator model"
    )
    forwarded = transport.json_bodies[1]["messages"][0]["content"]
    assert forwarded == "Explain this", "the request was not actually translated"


async def test_translator_reuses_the_callers_credentials():
    transport = RecordingTransport(
        Canned(
            json_body={
                "content": [{"type": "text", "text": '<segs>\n<seg id="1">\nHi\n</seg>\n</segs>'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
    )
    settings = Settings(upstream_anthropic_url=UPSTREAM, memo_persist=False)
    app = create_app(settings, transport=transport)

    await post(
        app,
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]},
        headers={"authorization": "Bearer caller-token"},
    )

    assert transport.requests[0].headers["authorization"] == "Bearer caller-token"
