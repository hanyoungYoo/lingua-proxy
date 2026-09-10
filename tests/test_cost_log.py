"""Savings accounting.

The number that matters is dollars, not tokens: the translator model is priced
differently from the main model, so a token win can still be a money loss. The
log must be able to say "this did not pay off".
"""

from __future__ import annotations

import json

from lingua_proxy.codecs import Usage
from lingua_proxy.cost_log import (
    PRICES,
    CostLog,
    CostRow,
    ModelPrice,
    estimate_tokens,
    summarize,
)


def test_known_models_have_prices():
    for model in ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"):
        assert model in PRICES


def test_price_lookup_falls_back_for_unknown_models():
    from lingua_proxy.cost_log import price_for

    assert price_for("some-model-we-have-never-seen") is not None


def test_price_lookup_matches_versioned_names():
    from lingua_proxy.cost_log import price_for

    assert (
        price_for("claude-sonnet-4-6-20260101").input_per_mtok
        == PRICES["claude-sonnet-4-6"].input_per_mtok
    )


def test_cost_includes_input_and_output():
    price = ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)

    assert price.cost(usage) == 18.0


def test_cache_reads_are_cheaper_than_fresh_input():
    price = ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)
    fresh = price.cost(Usage(input_tokens=1_000_000))
    cached = price.cost(Usage(cache_read_input_tokens=1_000_000))

    assert cached < fresh
    assert cached == 0.3


def test_cache_writes_cost_more_than_fresh_input():
    price = ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)
    write = price.cost(Usage(cache_creation_input_tokens=1_000_000))

    assert write == 3.75


def test_row_reports_savings_when_translation_wins():
    row = CostRow(
        model="claude-sonnet-4-6",
        translator_model="claude-haiku-4-5",
        usage=Usage(input_tokens=1000, output_tokens=500),
        translator_usage=Usage(input_tokens=300, output_tokens=300),
        counterfactual_input=2500,
        counterfactual_output=1300,
    )

    assert row.baseline_cost > row.proxied_cost
    assert row.savings_ratio > 0
    assert row.verdict == "pays off"


def test_row_reports_a_loss_when_translation_costs_more():
    """A short prompt cannot recover the translator's own fee."""
    row = CostRow(
        model="claude-sonnet-4-6",
        translator_model="claude-haiku-4-5",
        usage=Usage(input_tokens=20, output_tokens=10),
        translator_usage=Usage(input_tokens=4000, output_tokens=4000),
        counterfactual_input=30,
        counterfactual_output=14,
    )

    assert row.proxied_cost > row.baseline_cost
    assert row.savings_ratio < 0
    assert row.verdict == "loses"


def test_translator_cost_is_counted_against_savings():
    common = {
        "model": "claude-sonnet-4-6",
        "translator_model": "claude-haiku-4-5",
        "usage": Usage(input_tokens=1000, output_tokens=500),
        "counterfactual_input": 2500,
        "counterfactual_output": 1300,
    }
    free = CostRow(**common, translator_usage=Usage())
    paid = CostRow(**common, translator_usage=Usage(input_tokens=2000, output_tokens=2000))

    assert paid.savings_ratio < free.savings_ratio


def test_passthrough_row_has_no_savings_and_no_cost_delta():
    row = CostRow(model="claude-sonnet-4-6", usage=Usage(input_tokens=100, output_tokens=50))

    assert row.savings_ratio == 0.0
    assert row.verdict == "passthrough"


def test_estimate_tokens_scales_with_script():
    """Non-Latin scripts cost more tokens per character; that is the premise."""
    english = estimate_tokens("Explain why this function is slow and how to fix it.")
    korean = estimate_tokens("이 함수가 왜 느린지 설명해 주고 고치는 방법을 알려줘.")

    assert english > 0 and korean > 0
    assert korean / max(
        len("이 함수가 왜 느린지 설명해 주고 고치는 방법을 알려줘."), 1
    ) > english / max(len("Explain why this function is slow and how to fix it."), 1)


def test_writes_a_row_to_jsonl(tmp_path):
    log = CostLog(path=tmp_path / "cost.jsonl")
    log.record(
        CostRow(
            model="claude-sonnet-4-6",
            source_lang="ko",
            usage=Usage(input_tokens=100, output_tokens=50),
        )
    )

    line = json.loads((tmp_path / "cost.jsonl").read_text().strip())
    assert line["model"] == "claude-sonnet-4-6"
    assert line["source_lang"] == "ko"
    assert line["input_tokens"] == 100
    assert "ts" in line


def test_log_never_contains_prompt_text(tmp_path):
    """The cost log is shareable; the memo is not. Keep text out of it."""
    log = CostLog(path=tmp_path / "cost.jsonl")
    log.record(
        CostRow(
            model="claude-sonnet-4-6",
            source_lang="ko",
            usage=Usage(input_tokens=100, output_tokens=50),
            english_texts=["a secret prompt"],
            translated_texts=["비밀 프롬프트"],
        )
    )

    raw = (tmp_path / "cost.jsonl").read_text()
    assert "secret" not in raw
    assert "비밀" not in raw


def test_log_file_permissions_are_owner_only(tmp_path):
    path = tmp_path / "cost.jsonl"
    CostLog(path=path).record(CostRow(model="m", usage=Usage()))

    assert (path.stat().st_mode & 0o077) == 0


def test_summary_aggregates_requests_and_dollars(tmp_path):
    path = tmp_path / "cost.jsonl"
    log = CostLog(path=path)
    for _ in range(3):
        log.record(
            CostRow(
                model="claude-sonnet-4-6",
                translator_model="claude-haiku-4-5",
                source_lang="ko",
                translated=True,
                usage=Usage(input_tokens=1000, output_tokens=500),
                translator_usage=Usage(input_tokens=200, output_tokens=200),
                counterfactual_input=2500,
                counterfactual_output=1300,
            )
        )
    log.record(CostRow(model="claude-sonnet-4-6", usage=Usage(input_tokens=100, output_tokens=50)))

    summary = summarize(log.rows())
    assert summary.requests == 4
    assert summary.translated == 3
    assert summary.passthrough == 1
    assert summary.dollars_saved > 0
    assert "ko" in summary.by_language


def test_summary_of_an_empty_log_is_safe(tmp_path):
    summary = summarize(CostLog(path=tmp_path / "missing.jsonl").rows())

    assert summary.requests == 0
    assert summary.dollars_saved == 0.0
    assert summary.savings_ratio == 0.0


def test_summary_skips_corrupt_lines(tmp_path):
    path = tmp_path / "cost.jsonl"
    path.write_text(json.dumps({"model": "m", "input_tokens": 10}) + "\n" + '{"broken')

    summary = summarize(CostLog(path=path).rows())
    assert summary.requests == 1


# -- integration through the proxy --------------------------------------


async def test_a_translated_request_writes_a_cost_row(tmp_path):
    import httpx

    from lingua_proxy.config import Settings
    from lingua_proxy.proxy import create_app
    from tests.conftest import Canned, FakeTranslator, RecordingTransport

    korean = "이 함수가 왜 느린지 설명해 주고, 더 빠르게 만들 수 있는 방법을 알려줘."
    upstream_reply = {
        "id": "msg_1",
        "content": [{"type": "text", "text": "The function is slow because of the loop."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 120, "output_tokens": 60},
    }
    log_path = tmp_path / "cost.jsonl"
    settings = Settings(
        upstream_anthropic_url="https://gw.example/anthropic/",
        memo_persist=False,
        cost_log_path=log_path,
    )
    app = create_app(
        settings,
        transport=RecordingTransport(Canned(json_body=upstream_reply)),
        translator=FakeTranslator(),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as c:
        await c.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": korean}]},
        )

    row = json.loads(log_path.read_text().strip())
    assert row["translated"] is True
    assert row["source_lang"] == "ko"
    assert row["input_tokens"] == 120
    assert row["counterfactual_input"] > 0
    assert row["latency_ms"] >= 0


async def test_stats_endpoint_reports_the_summary(tmp_path):
    import httpx

    from lingua_proxy.config import Settings
    from lingua_proxy.proxy import create_app
    from tests.conftest import Canned, FakeTranslator, RecordingTransport

    log_path = tmp_path / "cost.jsonl"
    CostLog(path=log_path).record(
        CostRow(
            model="claude-sonnet-4-6",
            translator_model="claude-haiku-4-5",
            source_lang="ko",
            usage=Usage(input_tokens=1000, output_tokens=500),
            translator_usage=Usage(input_tokens=200, output_tokens=200),
            counterfactual_input=2500,
            counterfactual_output=1300,
        )
    )

    settings = Settings(
        upstream_anthropic_url="https://gw.example/anthropic/",
        memo_persist=False,
        cost_log_path=log_path,
    )
    app = create_app(
        settings,
        transport=RecordingTransport(Canned(json_body={})),
        translator=FakeTranslator(),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as c:
        resp = await c.get("/stats")

    body = resp.json()
    assert body["requests"] == 1
    assert body["translated"] == 1
    assert body["dollars_saved"] > 0
    assert body["by_language"] == {"ko": 1}
