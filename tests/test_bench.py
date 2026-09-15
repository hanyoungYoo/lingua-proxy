"""The benchmark must be capable of reporting bad news.

A savings tool that always claims savings is marketing. These tests pin the
arithmetic and check that losing categories are labelled as losses.
"""

from __future__ import annotations

import json
import pathlib

from lingua_proxy.bench import (
    BenchResult,
    build_report,
    load_corpus,
    verdict_for,
)


def test_corpus_ships_with_the_package():
    corpus = load_corpus()
    assert len(corpus) >= 15

    categories = {row["category"] for row in corpus}
    assert {"chat", "short", "code_heavy", "long_form"} <= categories


def test_corpus_covers_multiple_non_latin_scripts():
    langs = {row["lang"] for row in load_corpus()}
    assert {"ko", "ja", "zh"} <= langs


def test_corpus_entries_are_well_formed():
    for row in load_corpus():
        assert row["id"] and row["prompt"] and row["lang"] and row["category"]


def test_corpus_contains_no_endpoints_or_credentials():
    raw = pathlib.Path("lingua_proxy/bench_corpus.jsonl").read_text()
    assert "http://" not in raw
    assert "sk-" not in raw


def test_verdict_thresholds():
    assert verdict_for(0.55) == "pays off"
    assert verdict_for(0.25) == "pays off"
    assert verdict_for(0.10) == "marginal"
    assert verdict_for(0.0) == "loses"
    assert verdict_for(-0.7) == "loses"


def test_report_aggregates_by_category():
    results = [
        BenchResult("chat-1", "ko", "chat", baseline_cost=0.02, proxied_cost=0.01),
        BenchResult("chat-2", "ja", "chat", baseline_cost=0.02, proxied_cost=0.01),
        BenchResult("short-1", "ko", "short", baseline_cost=0.0005, proxied_cost=0.0009),
    ]
    report = build_report(results)

    chat = report["categories"]["chat"]
    assert chat["n"] == 2
    assert chat["savings_ratio"] == 0.5
    assert chat["verdict"] == "pays off"


def test_report_marks_a_losing_category_as_a_loss():
    results = [BenchResult("short-1", "ko", "short", baseline_cost=0.0005, proxied_cost=0.0009)]
    report = build_report(results)

    short = report["categories"]["short"]
    assert short["savings_ratio"] < 0
    assert short["verdict"] == "loses"


def test_report_overall_can_be_negative():
    results = [BenchResult("short-1", "ko", "short", baseline_cost=0.001, proxied_cost=0.004)]
    report = build_report(results)

    assert report["overall"]["savings_ratio"] < 0


def test_report_records_failures_without_crashing():
    results = [
        BenchResult("chat-1", "ko", "chat", baseline_cost=0.02, proxied_cost=0.01),
        BenchResult("chat-2", "ko", "chat", error="upstream refused"),
    ]
    report = build_report(results)

    assert report["categories"]["chat"]["n"] == 1
    assert report["errors"] == 1


def test_report_is_json_serializable():
    report = build_report(
        [BenchResult("chat-1", "ko", "chat", baseline_cost=0.02, proxied_cost=0.01)]
    )
    assert json.loads(json.dumps(report))["overall"]["n"] == 1


def test_empty_report_is_safe():
    report = build_report([])
    assert report["overall"]["n"] == 0
    assert report["overall"]["savings_ratio"] == 0.0


def test_exit_code_fails_when_chat_savings_are_below_the_floor():
    from lingua_proxy.bench import exit_code_for

    weak = build_report(
        [BenchResult("chat-1", "ko", "chat", baseline_cost=0.010, proxied_cost=0.0099)]
    )
    strong = build_report(
        [BenchResult("chat-1", "ko", "chat", baseline_cost=0.02, proxied_cost=0.01)]
    )

    assert exit_code_for(weak, min_savings=0.30) != 0
    assert exit_code_for(strong, min_savings=0.30) == 0


def test_exit_code_is_zero_when_no_chat_rows_were_measured():
    """Nothing measured is not the same as a failure."""
    from lingua_proxy.bench import exit_code_for

    report = build_report([BenchResult("x", "ko", "chat", error="skipped")])
    assert exit_code_for(report, min_savings=0.30) == 0


# -- recorded measurement -----------------------------------------------


def _recorded() -> dict:
    return json.loads(pathlib.Path("tests/fixtures/bench_recorded.json").read_text())


def test_recorded_measurement_is_committed():
    """The README's numbers must be traceable to a recorded run."""
    doc = _recorded()
    assert doc["measured_on"]
    assert doc["prompts"]
    assert doc["totals"]["input_savings_ratio"] > 0


def test_recorded_measurement_contains_no_endpoint_or_text():
    """The fixture is shareable: counts only, no prompts and no endpoints."""
    raw = pathlib.Path("tests/fixtures/bench_recorded.json").read_text()
    assert "http://" not in raw
    assert "Bearer" not in raw
    assert 'prompt"' not in raw.replace('"prompts"', "")


def test_recorded_totals_match_the_per_prompt_rows():
    doc = _recorded()
    native = sum(r["native_input_tokens"] for r in doc["prompts"])
    proxied = sum(r["proxied_input_tokens"] for r in doc["prompts"])

    assert doc["totals"]["native_input_tokens"] == native
    assert doc["totals"]["proxied_input_tokens"] == proxied


def test_recorded_short_prompts_show_no_savings():
    """Short prompts are deliberately not translated, so they must show 0%."""
    doc = _recorded()
    shorts = [r for r in doc["prompts"] if r["category"] == "short"]

    assert shorts
    assert all(r["input_savings_ratio"] == 0.0 for r in shorts)


def _head_to_head() -> dict:
    return json.loads(pathlib.Path("tests/fixtures/head_to_head.json").read_text())


def test_head_to_head_fixture_is_committed():
    doc = _head_to_head()
    assert doc["cases"]
    assert doc["main_model"] and doc["translator_model"]


def test_true_fee_measurement_is_the_one_the_readme_quotes():
    """Earlier runs estimated the translator fee; this run measured it.

    The README must quote the measured figures, not the superseded estimates.
    """
    doc = _head_to_head()["true_fee_measurement"]
    assert len(doc["cases"]) == 6
    readme = pathlib.Path("README.md").read_text()

    for case in doc["cases"]:
        pct = round(abs(case["saved"]) * 100)
        if case["saved"] > 0:
            expected = f"{pct}% cheaper"
        elif case["saved"] < 0:
            expected = f"{pct}% more expensive"
        else:
            expected = "| 0% |"
        assert expected in readme, f"README does not quote {expected} for {case}"


def test_true_fee_cases_charge_the_translator():
    """Every case carries real Haiku usage, not zeros or estimates."""
    doc = _head_to_head()["true_fee_measurement"]
    for case in doc["cases"]:
        assert case["haiku_in"] > 0 and case["haiku_out"] > 0


def test_formatting_off_pays_off_modestly_in_every_language():
    """The honest claim: 11-36% with formatting off. Bind it to the data."""
    doc = _head_to_head()["true_fee_measurement"]
    off = [c for c in doc["cases"] if not c["preserve_formatting"]]
    assert off and all(c["saved"] > 0 for c in off)
    assert min(c["saved"] for c in off) >= 0.10
    assert max(c["saved"] for c in off) <= 0.40


def test_break_even_predicts_the_measured_outcomes():
    """The pricing model explains the formatting-off data.

    With formatting on, the translated reply is longer than the native one,
    which the simple model does not account for; those cases are excluded
    here and the limitation is documented in the README.
    """
    from lingua_proxy.cost_log import break_even_shrink, price_for

    doc = _head_to_head()["true_fee_measurement"]
    threshold = break_even_shrink(price_for(doc["main_model"]), price_for(doc["translator_model"]))

    for case in doc["cases"]:
        if case["preserve_formatting"]:
            continue
        shrink = 1 - case["sonnet_english_out"] / case["sonnet_native_out"]
        assert (shrink >= threshold) == (case["saved"] > 0), case


def test_head_to_head_contains_no_endpoint_or_prompt_text():
    raw = pathlib.Path("tests/fixtures/head_to_head.json").read_text()
    assert "http" not in raw
    assert "Bearer" not in raw


def test_readme_quotes_the_recorded_overall_figure():
    """Stops the documented number from drifting away from the measurement."""
    doc = _recorded()
    figure = f"{doc['totals']['input_savings_ratio'] * 100:.1f}%"
    readme = pathlib.Path("README.md").read_text()

    assert figure in readme, f"README does not quote the measured figure {figure}"


def test_summarization_control_is_recorded():
    """The compression claim must be traceable to a controlled measurement."""
    doc = _head_to_head()
    control = doc["summarization_control"]

    # English delivered more text using fewer tokens: compression, not summary.
    assert control["english"]["chars"] > control["korean"]["chars"]
    assert control["english"]["output_tokens"] < control["korean"]["output_tokens"]


def test_readme_admits_the_content_loss():
    """The flattening cost is real and must not be quietly dropped."""
    readme = pathlib.Path("README.md").read_text()

    assert "summarization in disguise" in readme.lower()
    assert "flatter" in readme.lower()


# -- the bench must charge for the translator ---------------------------


def test_live_bench_includes_the_translator_fee(tmp_path, monkeypatch):
    """A Sonnet-only comparison reports a win on requests that lost money."""
    import json

    from lingua_proxy.bench_runner import run_live_bench
    from tests.conftest import Canned, RecordingTransport

    def seg(text, in_tok, out_tok):
        return Canned(
            json_body={
                "content": [
                    {"type": "text", "text": f'<segs>\n<seg id="1">\n{text}\n</seg>\n</segs>'}
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
            }
        )

    def main(out_tok):
        return Canned(
            json_body={
                "id": "m",
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": out_tok},
            }
        )

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "p1",
                "lang": "ko",
                "category": "chat",
                "prompt": "이 함수가 왜 느린지 설명해 주고, 더 빠르게 만들 수 있는 방법을 알려줘.",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        "lingua_proxy.bench_runner.load_corpus", lambda: [json.loads(corpus.read_text())]
    )
    monkeypatch.setenv("LINGUA_BENCH_AUTH", "t")

    # Native reply: 100 tokens. Proxied: Sonnet writes 90 (tiny shrink) but
    # Haiku then writes 500 tokens of Korean. Net must be a loss.
    transport = RecordingTransport(
        [main(100), seg("Explain", 40, 8), main(90), seg("긴 한국어 답변", 45, 500)]
    )

    report = run_live_bench(upstream="https://gw.example/anthropic/", transport=transport)
    bucket = report["categories"]["chat"]

    assert bucket["translator_cost"] > 0, "translator fee was not charged"
    assert bucket["proxied_cost"] > bucket["baseline_cost"]
    assert bucket["verdict"] == "loses"


# -- the measurements doc ------------------------------------------------
#
# The README quotes conclusions; docs/measurements.md carries the method and
# the raw counts. Both are prose about committed fixtures, so both can drift
# away from the data. These pin the doc the same way the README is pinned.


def _measurements() -> str:
    return pathlib.Path("docs/measurements.md").read_text()


def test_measurements_doc_exists_and_readme_links_to_it():
    """The README delegates the method; the link must not rot."""
    doc = _measurements()
    assert doc.strip()

    readme = pathlib.Path("README.md").read_text()
    assert "docs/measurements.md" in readme


def test_measurements_doc_quotes_every_head_to_head_case():
    """Every measured case appears in the doc, with its real token counts."""
    cases = _head_to_head()["true_fee_measurement"]["cases"]
    doc = _measurements()

    for case in cases:
        for field in ("sonnet_native_out", "sonnet_english_out", "haiku_in", "haiku_out"):
            assert str(case[field]) in doc, f"measurements doc omits {field}={case[field]}"


def test_measurements_doc_quotes_the_per_language_input_figures():
    """The per-language breakdown moved out of the README; pin it here."""
    by_lang = _recorded()["by_language"]
    doc = _measurements()

    for lang, row in by_lang.items():
        figure = f"{row['input_savings_ratio'] * 100:.1f}%"
        assert figure in doc, f"measurements doc omits {figure} for {lang}"


def test_measurements_doc_reports_the_truncation_caveat():
    """Four of six replies hit max_tokens. Burying that would flatter the tool."""
    doc = _measurements().lower()

    assert "max_tokens" in doc
    assert "four of" in doc, "the truncation count is not stated"


def test_measurements_doc_explains_the_superseded_estimate():
    """The correction is auditable: the doc must say what was wrong and why."""
    doc = _measurements().lower()

    assert "threefold" in doc
    assert "58" in doc and "69" in doc, "the superseded figures are not named"


def test_readme_keeps_the_verdict_not_just_the_method_link():
    """Delegating the method must not delegate the bad news with it.

    A reader who never opens the measurements doc still has to learn that
    savings can be zero, that truncated replies always lose, and that the
    honest range is 0-39%.
    """
    readme = pathlib.Path("README.md").read_text()

    assert "0–39%" in readme or "0-39%" in readme
    assert "max_tokens" in readme
    assert "44.6%" in readme
