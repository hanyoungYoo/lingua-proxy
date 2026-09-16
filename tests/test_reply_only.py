"""Code-heavy prompts keep the saving without paying to translate the request.

A prompt that is mostly code masks down to almost nothing before it reaches the
translator, so translating it costs a call and saves close to zero. Skipping
translation outright would also skip the *saving*, though: the model would
answer the Korean prompt in Korean, and the expensive model writing Korean is
the cost this proxy exists to avoid.

Reply-only mode forwards the user's own text with an instruction to answer in
English, and translates only the reply.
"""

from __future__ import annotations

import httpx

from lingua_proxy.config import Settings
from lingua_proxy.proxy import REPLY_IN_ENGLISH_INSTRUCTION, create_app
from lingua_proxy.segments import prose_share
from tests.conftest import Canned, FakeTranslator, RecordingTransport

UPSTREAM = "https://gw.example/anthropic/"

PROSE = "이 함수가 왜 느린지 설명해 주고, 더 빠르게 만들 수 있는 방법을 알려줘."

CODE_HEAVY = """다음 코드를 검토하고 성능 문제가 있는 부분을 설명해줘.

```python
def process(items):
    results = []
    for item in items:
        for other in items:
            if item.id == other.parent_id:
                results.append((item, other))
    return results


def summarize(rows):
    total = 0
    for row in rows:
        total += row.value
    return total / len(rows) if rows else 0
```
"""


def reply(text: str) -> dict:
    return {
        "id": "msg_1",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 40},
    }


def build(reply_text: str = "The nested loop is O(n^2).", **kw):
    transport = RecordingTransport(Canned(json_body=reply(reply_text)))
    translator = FakeTranslator()
    settings = Settings(upstream_anthropic_url=UPSTREAM, memo_persist=False, **kw)
    app = create_app(settings, transport=transport, translator=translator)
    return app, transport, translator


async def post(app, body, **kw):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as c:
        return await c.post("/v1/messages", json=body, **kw)


def ask(text: str) -> dict:
    return {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": text}]}


# -- the signal ---------------------------------------------------------


def test_prose_share_separates_prose_from_code():
    assert prose_share(PROSE) == 1.0
    assert prose_share(CODE_HEAVY) < 0.35
    assert prose_share("```python\nx = 1\n```") == 0.0


def test_prose_share_of_empty_text_is_zero():
    assert prose_share("") == 0.0
    assert prose_share("   ") == 0.0


async def test_a_prompt_whose_prose_is_too_short_to_detect_passes_through():
    """The detector's floor still comes first, and should.

    The most code-heavy prompts carry the least prose, so some fall below the
    minimum length at which detection is trustworthy. Those pass through
    untouched rather than being routed anywhere on a guess: a missed saving is
    a rounding error, a misdetected prompt is not.
    """
    app, _, translator = build()
    resp = await post(app, ask("코드 봐줘.\n\n```python\nx = 1\n```"))

    assert resp.headers.get("x-lingua-translated") == "false"
    assert translator.calls == []


# -- mode selection -----------------------------------------------------


async def test_code_heavy_prompt_does_not_pay_to_translate_the_request():
    app, _, translator = build()
    resp = await post(app, ask(CODE_HEAVY))

    assert resp.status_code == 200
    # The only translation call is the reply coming back.
    assert [(src, tgt) for _, src, tgt in translator.calls] == [("en", "ko")]


async def test_code_heavy_prompt_still_translates_the_reply():
    app, _, _ = build(reply_text="The nested loop is O(n^2).")
    resp = await post(app, ask(CODE_HEAVY))

    text = resp.json()["content"][0]["text"]
    assert text == "«ko»The nested loop is O(n^2)."


async def test_code_heavy_prompt_is_forwarded_in_the_users_own_language():
    app, transport, _ = build()
    await post(app, ask(CODE_HEAVY))

    sent = transport.json_bodies[-1]
    assert sent["messages"][-1]["content"] == CODE_HEAVY


async def test_code_heavy_prompt_asks_the_model_to_answer_in_english():
    app, transport, _ = build()
    await post(app, ask(CODE_HEAVY))

    sent = transport.json_bodies[-1]
    assert REPLY_IN_ENGLISH_INSTRUCTION in str(sent.get("system"))


async def test_prose_prompt_still_translates_the_request():
    app, _, translator = build()
    await post(app, ask(PROSE))

    directions = [(src, tgt) for _, src, tgt in translator.calls]
    assert ("ko", "en") in directions


async def test_threshold_of_zero_disables_the_mode():
    app, _, translator = build(reply_only_prose_share=0.0)
    await post(app, ask(CODE_HEAVY))

    directions = [(src, tgt) for _, src, tgt in translator.calls]
    assert ("ko", "en") in directions


async def test_a_high_threshold_routes_ordinary_prose_through_reply_only():
    app, _, translator = build(reply_only_prose_share=1.1)
    await post(app, ask(PROSE))

    assert [(src, tgt) for _, src, tgt in translator.calls] == [("en", "ko")]


# -- per-request override -----------------------------------------------


async def test_header_can_force_reply_only_on():
    app, _, translator = build()
    await post(app, ask(PROSE), headers={"x-lingua-reply-only": "true"})

    assert [(src, tgt) for _, src, tgt in translator.calls] == [("en", "ko")]


async def test_header_can_force_reply_only_off():
    app, _, translator = build()
    await post(app, ask(CODE_HEAVY), headers={"x-lingua-reply-only": "false"})

    directions = [(src, tgt) for _, src, tgt in translator.calls]
    assert ("ko", "en") in directions


# -- transparency -------------------------------------------------------


async def test_reply_only_is_announced_in_the_response():
    app, _, _ = build()
    resp = await post(app, ask(CODE_HEAVY))

    assert resp.headers.get("x-lingua-reply-only") == "true"
    assert resp.headers.get("x-lingua-translated") == "true"
    assert resp.headers.get("x-lingua-source-lang") == "ko"


async def test_reply_only_does_not_claim_an_english_prompt_it_never_sent():
    """The echo header reports the English the model received.

    In reply-only mode there is none, and reporting the user's own Korean
    under that header would be a lie.
    """
    app, _, _ = build()
    resp = await post(app, ask(CODE_HEAVY))

    assert "x-lingua-prompt-en" not in resp.headers


async def test_ordinary_translation_is_not_marked_reply_only():
    app, _, _ = build()
    resp = await post(app, ask(PROSE))

    assert "x-lingua-reply-only" not in resp.headers


# -- fail open ----------------------------------------------------------


async def test_a_reply_that_ignored_the_instruction_is_left_alone():
    """The model was asked for English and answered in Korean anyway.

    Translating now would round-trip the user's own language for nothing, so
    the reply is handed back as it came.
    """
    korean_reply = "이 중첩 루프는 O(n^2)입니다. 딕셔너리를 쓰면 더 빠릅니다."
    app, _, translator = build(reply_text=korean_reply)
    resp = await post(app, ask(CODE_HEAVY))

    assert resp.json()["content"][0]["text"] == korean_reply
    assert translator.calls == []


async def test_an_english_reply_quoting_the_user_is_still_translated():
    """A little of the user's script in an English answer is not a failure."""
    app, _, translator = build(
        reply_text="The nested loop is O(n^2). You asked about 처리 speed specifically."
    )
    resp = await post(app, ask(CODE_HEAVY))

    assert [(src, tgt) for _, src, tgt in translator.calls] == [("en", "ko")]
    assert resp.json()["content"][0]["text"].startswith("«ko»")


# -- honest accounting --------------------------------------------------


async def test_reply_only_does_not_book_an_input_saving_it_did_not_make(tmp_path):
    """The prompt went up in Korean, so there is no input-side saving.

    The counterfactual for a reply-only request is the request itself. Booking
    a saving here would report a number the user never received.
    """
    import json

    log_path = tmp_path / "cost.jsonl"
    transport = RecordingTransport(Canned(json_body=reply("The nested loop is O(n^2).")))
    settings = Settings(
        upstream_anthropic_url=UPSTREAM,
        memo_persist=False,
        cost_log_path=log_path,
        preserve_formatting=False,
    )
    app = create_app(settings, transport=transport, translator=FakeTranslator())
    await post(app, ask(CODE_HEAVY))

    row = json.loads(log_path.read_text().strip())
    total_input = (
        row["input_tokens"] + row["cache_read_input_tokens"] + row["cache_creation_input_tokens"]
    )
    assert row["counterfactual_input"] == total_input, (
        "reply-only booked an input saving, but the prompt went up untranslated"
    )


# -- byte stability -----------------------------------------------------


async def test_reply_only_leaves_history_reproducible_on_the_next_turn():
    """Nothing is memoised for the request, so the bytes resend unchanged.

    An agentic client resends the whole conversation each turn. If reply-only
    mode wrote a translation into the memo, the next turn would rewrite this
    turn's user text and miss the upstream prompt cache.
    """
    app, transport, _ = build()
    await post(app, ask(CODE_HEAVY))

    follow_up = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": CODE_HEAVY},
            {"role": "assistant", "content": "«ko»The nested loop is O(n^2)."},
            {"role": "user", "content": CODE_HEAVY},
        ],
    }
    await post(app, follow_up)

    sent = transport.json_bodies[-1]
    assert sent["messages"][0]["content"] == CODE_HEAVY
