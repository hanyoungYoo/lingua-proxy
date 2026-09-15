"""The translator's own cost must be measured, not assumed to be zero.

Every cost row and every bench result compared the expensive model's usage on
both paths and silently omitted the fee for the cheap model that wrote the
user's language. That fee is ~92% of the round trip's overhead, so omitting it
overstates savings and can report a win on a request that lost money.
"""

from __future__ import annotations

import httpx

from lingua_proxy.config import Settings
from lingua_proxy.proxy import create_app
from lingua_proxy.translator import LLMTranslator
from tests.conftest import Canned, RecordingTransport

UPSTREAM = "https://gw.example/anthropic/"
KOREAN = "이 함수가 왜 느린지 설명해 주고, 더 빠르게 만들 수 있는 방법을 알려줘."


def seg_reply(text: str, *, in_tok: int, out_tok: int) -> dict:
    return {
        "content": [{"type": "text", "text": f'<segs>\n<seg id="1">\n{text}\n</seg>\n</segs>'}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def main_reply(text: str) -> dict:
    return {
        "id": "msg_1",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 40},
    }


async def test_translator_reports_the_usage_the_api_returned():
    transport = RecordingTransport(Canned(json_body=seg_reply("Hello", in_tok=37, out_tok=11)))
    translator = LLMTranslator(upstream_url=UPSTREAM, client=httpx.AsyncClient(transport=transport))

    await translator.translate(["안녕하세요"], "ko", "en")

    assert translator.last_usage.input_tokens == 37
    assert translator.last_usage.output_tokens == 11


async def test_translator_usage_accumulates_across_calls():
    transport = RecordingTransport(
        [
            Canned(json_body=seg_reply("A", in_tok=10, out_tok=5)),
            Canned(json_body=seg_reply("B", in_tok=20, out_tok=7)),
        ]
    )
    translator = LLMTranslator(upstream_url=UPSTREAM, client=httpx.AsyncClient(transport=transport))

    await translator.translate(["하나"], "ko", "en")
    await translator.translate(["둘"], "ko", "en")

    assert translator.total_usage.input_tokens == 30
    assert translator.total_usage.output_tokens == 12


async def test_cost_row_records_the_translators_real_usage(tmp_path):
    """Both legs: prompt in (call 1) and reply back (call 3)."""
    import json

    transport = RecordingTransport(
        [
            Canned(json_body=seg_reply("Explain this", in_tok=40, out_tok=8)),
            Canned(json_body=main_reply("It is slow because of the loop.")),
            Canned(json_body=seg_reply("루프 때문에 느립니다", in_tok=25, out_tok=30)),
        ]
    )
    log_path = tmp_path / "cost.jsonl"
    settings = Settings(
        upstream_anthropic_url=UPSTREAM,
        memo_persist=False,
        cost_log_path=log_path,
        preserve_formatting=False,
    )
    app = create_app(settings, transport=transport)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as c:
        await c.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]},
            headers={"authorization": "Bearer t"},
        )

    row = json.loads(log_path.read_text().strip())
    assert row["translator_input_tokens"] == 65, "translator input usage was not recorded"
    assert row["translator_output_tokens"] == 38, "translator output usage was not recorded"
    assert row["translator_calls"] == 2


async def test_a_request_that_loses_money_on_the_fee_is_reported_as_a_loss(tmp_path):
    """Sonnet saves a little; Haiku's fee eats more than that. Net must be negative."""

    from lingua_proxy.cost_log import CostLog

    transport = RecordingTransport(
        [
            Canned(json_body=seg_reply("Explain", in_tok=40, out_tok=8)),
            # Expensive model barely shrank: 40 English tokens vs a 44-token
            # counterfactual. A fee of 400 Haiku output tokens dwarfs that.
            Canned(
                json_body={
                    "id": "m",
                    "content": [{"type": "text", "text": "short"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 30, "output_tokens": 40},
                }
            ),
            Canned(json_body=seg_reply("짧은 답변", in_tok=45, out_tok=400)),
        ]
    )
    log_path = tmp_path / "cost.jsonl"
    settings = Settings(
        upstream_anthropic_url=UPSTREAM,
        memo_persist=False,
        cost_log_path=log_path,
        preserve_formatting=False,
    )
    app = create_app(settings, transport=transport)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as c:
        await c.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": KOREAN}]},
            headers={"authorization": "Bearer t"},
        )

    (row,) = CostLog(path=log_path).rows()
    # Both legs accumulate: 8 tokens translating the prompt in, 400 writing
    # the Korean reply back.
    assert row.translator_usage.output_tokens == 408
    assert row.proxied_cost > row.baseline_cost, "the fee was omitted from the comparison"
    assert row.verdict == "loses"
