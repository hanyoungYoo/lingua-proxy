"""CLI behaviour, especially the transactional settings patching.

`wrap` edits a file that belongs to another tool. The contract is: back out
cleanly on every exit path, never clobber a file we cannot parse, and never
hard-code anyone's gateway.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from lingua_proxy.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch) -> pathlib.Path:
    work = tmp_path / "project"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def local_settings(project: pathlib.Path) -> dict:
    return json.loads((project / ".claude" / "settings.local.json").read_text())


def test_version_is_reported(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "lingua-proxy" in result.output


def test_help_lists_every_command(runner):
    result = runner.invoke(main, ["--help"])
    for command in ("proxy", "wrap", "unwrap", "doctor", "dashboard", "bench"):
        assert command in result.output


# -- wrap ---------------------------------------------------------------


def test_wrap_patches_project_settings_and_restores_on_exit(runner, project, isolated_home):
    result = runner.invoke(main, ["wrap", "claude", "--no-proxy", "--dry-run", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:9999" in result.output
    # dry-run restores immediately, so nothing is left behind
    assert not (project / ".claude" / "settings.local.json").exists()


def test_wrap_chains_onto_an_existing_gateway(runner, project, isolated_home):
    """A user already behind a gateway must keep using it as our upstream."""
    claude_dir = isolated_home / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.json").write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "https://gw.example:8443/anthropic/",
                    "ANTHROPIC_AUTH_TOKEN": "existing-token",
                }
            }
        )
    )

    result = runner.invoke(main, ["wrap", "claude", "--no-proxy", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "https://gw.example:8443/anthropic/" in result.output, (
        "the existing gateway was not detected as the upstream"
    )


def test_wrap_persists_the_resolved_upstream_for_the_proxy(runner, project, isolated_home):
    claude_dir = isolated_home / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://gw.example/anthropic/"}})
    )

    runner.invoke(main, ["wrap", "claude", "--no-proxy", "--dry-run"])

    config = (isolated_home / ".lingua-proxy" / "config.toml").read_text()
    assert "https://gw.example/anthropic/" in config


def test_wrap_preserves_other_keys_in_project_settings(runner, project, isolated_home):
    claude_dir = project / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.local.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash(ls)"]}, "env": {"FOO": "bar"}})
    )

    runner.invoke(main, ["wrap", "claude", "--no-proxy", "--keep-patched"])

    patched = local_settings(project)
    assert patched["permissions"] == {"allow": ["Bash(ls)"]}
    assert patched["env"]["FOO"] == "bar"
    assert patched["env"]["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1")


def test_wrap_restores_a_previous_base_url_exactly(runner, project, isolated_home):
    claude_dir = project / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.local.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://previous.example/"}})
    )

    runner.invoke(main, ["wrap", "claude", "--no-proxy", "--dry-run"])

    assert local_settings(project)["env"]["ANTHROPIC_BASE_URL"] == "https://previous.example/"


def test_wrap_refuses_to_touch_unparseable_settings(runner, project, isolated_home):
    claude_dir = project / ".claude"
    claude_dir.mkdir()
    broken = claude_dir / "settings.local.json"
    broken.write_text("{ this is not json")

    result = runner.invoke(main, ["wrap", "claude", "--no-proxy", "--dry-run"])

    assert result.exit_code != 0
    assert broken.read_text() == "{ this is not json", "a file we could not parse was overwritten"


def test_wrap_refuses_to_nest(runner, project, isolated_home, monkeypatch):
    monkeypatch.setenv("LINGUA_PROXY_WRAPPED", "1")
    result = runner.invoke(main, ["wrap", "claude", "--no-proxy", "--dry-run"])

    assert result.exit_code != 0
    assert "already" in result.output.lower()


def test_wrap_reports_a_missing_claude_binary(runner, project, isolated_home, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = runner.invoke(main, ["wrap", "claude", "--no-proxy"])

    assert result.exit_code != 0
    assert "claude" in result.output.lower()


def test_wrap_explicit_upstream_flag_wins(runner, project, isolated_home):
    claude_dir = isolated_home / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://ignored.example/"}})
    )

    result = runner.invoke(
        main,
        ["wrap", "claude", "--no-proxy", "--dry-run", "--upstream", "https://chosen.example/"],
    )

    assert "https://chosen.example/" in result.output
    assert "https://ignored.example/" not in result.output


# -- unwrap -------------------------------------------------------------


def test_unwrap_restores_after_a_crashed_wrap(runner, project, isolated_home):
    runner.invoke(main, ["wrap", "claude", "--no-proxy", "--keep-patched"])
    assert "127.0.0.1" in local_settings(project)["env"]["ANTHROPIC_BASE_URL"]

    result = runner.invoke(main, ["unwrap", "claude", "--no-stop-proxy"])

    assert result.exit_code == 0, result.output
    assert not (project / ".claude" / "settings.local.json").exists()


def test_unwrap_is_idempotent(runner, project, isolated_home):
    first = runner.invoke(main, ["unwrap", "claude", "--no-stop-proxy"])
    second = runner.invoke(main, ["unwrap", "claude", "--no-stop-proxy"])

    assert first.exit_code == 0
    assert second.exit_code == 0


def test_unwrap_leaves_unrelated_settings_alone(runner, project, isolated_home):
    claude_dir = project / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.local.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash(ls)"]}})
    )

    runner.invoke(main, ["wrap", "claude", "--no-proxy", "--keep-patched"])
    runner.invoke(main, ["unwrap", "claude", "--no-stop-proxy"])

    assert local_settings(project)["permissions"] == {"allow": ["Bash(ls)"]}


# -- doctor -------------------------------------------------------------


def test_doctor_reports_checks_and_exits_nonzero_when_proxy_is_down(runner, isolated_home):
    result = runner.invoke(main, ["doctor"])

    assert "Python" in result.output
    assert result.exit_code != 0, "a dead proxy should be a failure"


def test_doctor_json_output_is_machine_readable(runner, isolated_home):
    result = runner.invoke(main, ["doctor", "--json"])
    payload = json.loads(result.output)

    assert "checks" in payload
    assert all("name" in c and "status" in c for c in payload["checks"])


def test_doctor_never_prints_a_credential(runner, isolated_home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "super-secret-value")
    result = runner.invoke(main, ["doctor"])

    assert "super-secret-value" not in result.output


# -- dashboard ----------------------------------------------------------


def test_dashboard_on_an_empty_log_is_friendly(runner, isolated_home):
    result = runner.invoke(main, ["dashboard"])

    assert result.exit_code == 0
    assert "no" in result.output.lower()


def test_dashboard_renders_savings(runner, isolated_home):
    from lingua_proxy.codecs import Usage
    from lingua_proxy.cost_log import CostLog, CostRow

    log = CostLog(path=isolated_home / ".lingua-proxy" / "cost_log.jsonl")
    log.record(
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

    result = runner.invoke(main, ["dashboard"])

    assert result.exit_code == 0
    assert "ko" in result.output
    assert "%" in result.output


def test_dashboard_json_output(runner, isolated_home):
    result = runner.invoke(main, ["dashboard", "--json"])
    payload = json.loads(result.output)

    assert "requests" in payload
    assert "dollars_saved" in payload


def test_wrap_sets_the_base_url_on_the_child_process(runner, project, isolated_home, tmp_path):
    """The launched client must not inherit a stale base URL from the shell."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    script = fake_bin / "claude"
    dumped = tmp_path / "child-env.txt"
    script.write_text(
        f'#!/bin/sh\nprintf "%s\\n%s\\n" "$ANTHROPIC_BASE_URL" "$LINGUA_PROXY_WRAPPED" > {dumped}\n'
    )
    script.chmod(0o755)

    import shutil as _shutil

    original_which = _shutil.which
    try:
        _shutil.which = lambda name: str(script) if name == "claude" else original_which(name)
        result = runner.invoke(
            main,
            [
                "wrap",
                "claude",
                "--no-proxy",
                "--port",
                "8787",
                "--upstream",
                "https://gw.example/anthropic/",
            ],
        )
    finally:
        _shutil.which = original_which

    assert result.exit_code == 0, result.output
    base_url, wrapped = dumped.read_text().splitlines()
    assert base_url == "http://127.0.0.1:8787", "child inherited the wrong base URL"
    assert wrapped == "1"
