"""The translator calls a cheap model and must never corrupt a request.

Failure is always safe: any malformed reply raises TranslationError, and the
pipeline falls back to sending the user's original text through untranslated.
"""

from __future__ import annotations

import json

import httpx
import pytest

from lingua_proxy.translator import (
    DeepLTranslator,
    LibreTranslator,
    LLMTranslator,
    TranslationError,
)
from tests.conftest import Canned, RecordingTransport

UPSTREAM = "https://gw.example/anthropic/"


def reply(*segments: str, stop_reason: str = "end_turn") -> dict:
    body = (
        "<segs>\n"
        + "\n".join(f'<seg id="{i + 1}">\n{seg}\n</seg>' for i, seg in enumerate(segments))
        + "\n</segs>"
    )
    return {
        "id": "msg_t",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": body}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }


def make(script, **kw) -> tuple[LLMTranslator, RecordingTransport]:
    transport = RecordingTransport(script)
    client = httpx.AsyncClient(transport=transport)
    translator = LLMTranslator(upstream_url=UPSTREAM, client=client, **kw)
    return translator, transport


async def test_translates_a_batch_in_one_call():
    translator, transport = make(Canned(json_body=reply("Explain this", "And this")))
    out = await translator.translate(["이것을 설명해", "그리고 이것도"], "ko", "en")

    assert out == ["Explain this", "And this"]
    assert len(transport.requests) == 1, "segments must batch into a single call"


async def test_request_shape_is_deterministic_and_cheap():
    translator, transport = make(Canned(json_body=reply("Hello")))
    await translator.translate(["안녕하세요"], "ko", "en")

    body = transport.json_bodies[0]
    assert body["model"] == "claude-haiku-4-5"
    assert body["temperature"] == 0
    assert "thinking" not in body
    assert body["max_tokens"] > 0
    assert '<seg id="1">' in body["messages"][0]["content"]
    assert "안녕하세요" in body["messages"][0]["content"]


async def test_system_prompt_names_both_languages():
    translator, transport = make(Canned(json_body=reply("Hello")))
    await translator.translate(["안녕하세요"], "ko", "en")

    system = transport.json_bodies[0]["system"]
    text = system if isinstance(system, str) else json.dumps(system)
    assert "ko" in text.lower() and "en" in text.lower()


async def test_placeholders_are_passed_through_untouched():
    translator, transport = make(Canned(json_body=reply("Fix <lp0/> please")))
    (out,) = await translator.translate(["<lp0/> 를 고쳐줘"], "ko", "en")

    assert "<lp0/>" in transport.json_bodies[0]["messages"][0]["content"]
    assert out == "Fix <lp0/> please"


async def test_parser_tolerates_preamble_and_code_fence():
    messy = {
        "content": [
            {
                "type": "text",
                "text": 'Sure, here you go:\n```\n<segs>\n<seg id="1">\nHello\n</seg>\n</segs>\n```',
            }
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    translator, _ = make(Canned(json_body=messy))
    assert await translator.translate(["안녕"], "ko", "en") == ["Hello"]


async def test_multiline_segment_preserves_internal_newlines():
    translator, _ = make(Canned(json_body=reply("line one\nline two")))
    (out,) = await translator.translate(["첫 줄\n둘째 줄"], "ko", "en")
    assert out == "line one\nline two"


async def test_missing_segment_id_raises():
    translator, _ = make(Canned(json_body=reply("only one")))
    with pytest.raises(TranslationError):
        await translator.translate(["하나", "둘"], "ko", "en")


async def test_unparseable_output_raises():
    bad = {
        "content": [{"type": "text", "text": "I cannot do that."}],
        "stop_reason": "end_turn",
        "usage": {},
    }
    translator, _ = make(Canned(json_body=bad))
    with pytest.raises(TranslationError):
        await translator.translate(["안녕"], "ko", "en")


async def test_truncated_output_raises():
    """A max_tokens stop means the last segment is probably cut in half."""
    translator, _ = make(Canned(json_body=reply("Hello", stop_reason="max_tokens")))
    with pytest.raises(TranslationError):
        await translator.translate(["안녕"], "ko", "en")


async def test_refusal_raises():
    translator, _ = make(Canned(json_body=reply("Hello", stop_reason="refusal")))
    with pytest.raises(TranslationError):
        await translator.translate(["안녕"], "ko", "en")


async def test_http_error_raises_translation_error():
    translator, _ = make(Canned(status=400, json_body={"error": {"message": "bad"}}))
    with pytest.raises(TranslationError):
        await translator.translate(["안녕"], "ko", "en")


async def test_rate_limit_is_retried_once_then_succeeds():
    translator, transport = make(
        [Canned(status=429, json_body={}), Canned(json_body=reply("Hello"))]
    )
    assert await translator.translate(["안녕"], "ko", "en") == ["Hello"]
    assert len(transport.requests) == 2


async def test_persistent_rate_limit_raises_after_the_retry():
    translator, transport = make(Canned(status=429, json_body={}))
    with pytest.raises(TranslationError):
        await translator.translate(["안녕"], "ko", "en")
    assert len(transport.requests) == 2, "exactly one retry, then give up"


async def test_connection_error_raises_translation_error():
    translator, _ = make(Canned(exc=httpx.ConnectError("refused")))
    with pytest.raises(TranslationError):
        await translator.translate(["안녕"], "ko", "en")


async def test_caller_credentials_are_reused():
    translator, transport = make(
        Canned(json_body=reply("Hello")),
        auth_headers={"authorization": "Bearer caller-token", "anthropic-version": "2023-06-01"},
    )
    await translator.translate(["안녕"], "ko", "en")

    sent = transport.requests[0].headers
    assert sent["authorization"] == "Bearer caller-token"
    assert sent["anthropic-version"] == "2023-06-01"


async def test_only_the_oauth_beta_is_forwarded():
    """Other betas change validation of an unrelated request, so they are dropped."""
    translator, transport = make(
        Canned(json_body=reply("Hello")),
        auth_headers={
            "authorization": "Bearer t",
            "anthropic-beta": "oauth-2025-04-20,fast-mode-2026-02-01,compact-2026-01-12",
        },
    )
    await translator.translate(["안녕"], "ko", "en")

    beta = transport.requests[0].headers.get("anthropic-beta", "")
    assert beta == "oauth-2025-04-20"


async def test_claude_code_headers_are_never_forwarded_to_the_translator():
    translator, transport = make(
        Canned(json_body=reply("Hello")),
        auth_headers={"authorization": "Bearer t", "x-claude-code-session-id": "sess-1"},
    )
    await translator.translate(["안녕"], "ko", "en")
    assert "x-claude-code-session-id" not in transport.requests[0].headers


async def test_explicit_api_key_overrides_caller_credentials():
    translator, transport = make(
        Canned(json_body=reply("Hello")),
        auth_headers={"authorization": "Bearer caller-token"},
        api_key="sk-dedicated",
    )
    await translator.translate(["안녕"], "ko", "en")

    sent = transport.requests[0].headers
    assert sent["x-api-key"] == "sk-dedicated"
    assert "authorization" not in sent


async def test_empty_segment_list_makes_no_call():
    translator, transport = make(Canned(json_body=reply()))
    assert await translator.translate([], "ko", "en") == []
    assert transport.requests == []


async def test_large_batch_is_split_across_calls():
    segments = [f"긴 문장 {i} " + "가" * 400 for i in range(12)]
    script = [Canned(json_body=reply(*[f"long {i}" for i in range(6)])) for _ in range(4)]
    translator, transport = make(script, max_batch_chars=2000)
    out = await translator.translate(segments, "ko", "en")

    assert len(transport.requests) > 1, "an oversized batch must be split"
    assert len(out) == len(segments)


def test_alternative_backends_are_declared_but_not_implemented():
    """v0.1 ships the interface only, so the shape is fixed before adapters land."""
    for cls in (DeepLTranslator, LibreTranslator):
        with pytest.raises(NotImplementedError):
            cls()
