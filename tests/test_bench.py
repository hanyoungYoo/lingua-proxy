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


def test_readme_quotes_the_recorded_overall_figure():
    """Stops the documented number from drifting away from the measurement."""
    doc = _recorded()
    figure = f"{doc['totals']['input_savings_ratio'] * 100:.1f}%"
    readme = pathlib.Path("README.md").read_text()

    assert figure in readme, f"README does not quote the measured figure {figure}"
