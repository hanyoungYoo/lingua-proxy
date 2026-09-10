"""`lingua-proxy advise` answers "will this configuration pay off?"

The break-even condition is arithmetic, not something a user should have to
discover by spending money on a sweep.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from lingua_proxy.cli import main


def run(*args):
    return CliRunner().invoke(main, ["advise", *args])


def test_recommends_the_cheapest_translator_for_a_given_model():
    result = run("--model", "claude-sonnet-4-6")

    assert result.exit_code == 0
    assert "haiku" in result.output.lower()


def test_reports_the_required_shrink():
    result = run("--model", "claude-sonnet-4-6", "--json")
    payload = json.loads(result.output)

    best = payload["recommended"]
    assert 0.30 < best["required_shrink"] < 0.45


def test_warns_when_the_main_model_is_too_cheap_to_benefit():
    result = run("--model", "claude-haiku-4-5")

    assert result.exit_code != 0
    assert "not worth" in result.output.lower() or "never" in result.output.lower()


def test_opus_needs_less_shrink_than_sonnet():
    opus = json.loads(run("--model", "claude-opus-4-8", "--json").output)
    sonnet = json.loads(run("--model", "claude-sonnet-4-6", "--json").output)

    assert opus["recommended"]["required_shrink"] < sonnet["recommended"]["required_shrink"]


def test_lists_every_candidate_translator_with_its_verdict():
    payload = json.loads(run("--model", "claude-sonnet-4-6", "--json").output)

    assert len(payload["candidates"]) >= 3
    assert all("required_shrink" in c for c in payload["candidates"])
    assert any(c["viable"] is False for c in payload["candidates"])


def test_unknown_model_still_produces_advice():
    """Falls back to default pricing rather than refusing to answer."""
    result = run("--model", "some-unreleased-model")
    assert result.exit_code == 0
