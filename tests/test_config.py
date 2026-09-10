"""Upstream resolution must be generic: configured, never hard-coded."""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from lingua_proxy.config import Settings, resolve_anthropic_upstream, resolve_openai_upstream

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_defaults_are_the_public_apis():
    s = Settings()
    assert s.upstream_anthropic_url == "https://api.anthropic.com"
    assert s.upstream_openai_url == "https://api.openai.com"
    assert s.host == "127.0.0.1"
    assert s.port == 8787


def test_flag_beats_env_beats_toml_beats_client_settings(tmp_path, monkeypatch):
    toml = tmp_path / "config.toml"
    toml.write_text('upstream_anthropic_url = "https://from-toml.example/anthropic/"\n')
    claude = tmp_path / "settings.json"
    claude.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://from-client.example/"}}))

    # lowest precedence: the client's own configured base URL
    assert (
        resolve_anthropic_upstream(
            config_path=tmp_path / "missing.toml", client_settings_path=claude
        )
        == "https://from-client.example/"
    )

    # config file beats it
    assert (
        resolve_anthropic_upstream(config_path=toml, client_settings_path=claude)
        == "https://from-toml.example/anthropic/"
    )

    # env beats the config file
    monkeypatch.setenv("LINGUA_UPSTREAM_ANTHROPIC_URL", "https://from-env.example/")
    assert (
        resolve_anthropic_upstream(config_path=toml, client_settings_path=claude)
        == "https://from-env.example/"
    )

    # an explicit flag beats everything
    assert (
        resolve_anthropic_upstream(
            flag="https://from-flag.example/", config_path=toml, client_settings_path=claude
        )
        == "https://from-flag.example/"
    )


def test_falls_back_to_public_api_when_nothing_is_configured(tmp_path):
    assert (
        resolve_anthropic_upstream(
            config_path=tmp_path / "none.toml", client_settings_path=tmp_path / "none.json"
        )
        == "https://api.anthropic.com"
    )
    assert resolve_openai_upstream(config_path=tmp_path / "none.toml") == "https://api.openai.com"


def test_shell_env_base_url_is_used_when_client_settings_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://shell.example/anthropic/")
    assert (
        resolve_anthropic_upstream(
            config_path=tmp_path / "none.toml", client_settings_path=tmp_path / "none.json"
        )
        == "https://shell.example/anthropic/"
    )


def test_client_settings_beats_shell_env(tmp_path, monkeypatch):
    """Claude Code applies its settings env over the shell, so we mirror that."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://shell.example/")
    claude = tmp_path / "settings.json"
    claude.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://settings.example/"}}))
    assert (
        resolve_anthropic_upstream(config_path=tmp_path / "none.toml", client_settings_path=claude)
        == "https://settings.example/"
    )


def test_unparseable_client_settings_is_ignored_not_fatal(tmp_path):
    claude = tmp_path / "settings.json"
    claude.write_text("{ this is not json")
    assert (
        resolve_anthropic_upstream(config_path=tmp_path / "none.toml", client_settings_path=claude)
        == "https://api.anthropic.com"
    )


@pytest.mark.parametrize(
    "base,path,query,expected",
    [
        ("https://api.anthropic.com", "/v1/messages", "", "https://api.anthropic.com/v1/messages"),
        (
            "https://gw.example/anthropic/",
            "/v1/messages",
            "beta=true",
            "https://gw.example/anthropic/v1/messages?beta=true",
        ),
        (
            "https://gw.example:8443/prefix",
            "/v1/messages",
            "",
            "https://gw.example:8443/prefix/v1/messages",
        ),
        ("http://127.0.0.1:9999/", "/v1/models", "", "http://127.0.0.1:9999/v1/models"),
    ],
)
def test_upstream_url_join_preserves_prefix_port_and_query(base, path, query, expected):
    from lingua_proxy.config import join_upstream

    assert join_upstream(base, path, query) == expected


def test_no_private_endpoints_committed_in_repo():
    """Guard: a developer's private gateway must never be committed.

    Scans packaged source, tests and fixtures for non-public http(s) hosts.
    Loopback is allowed only as a bind address or in test URLs, and the public
    vendor APIs are allowed anywhere.
    """
    allowed_hosts = {
        "api.anthropic.com",
        "api.openai.com",
        "127.0.0.1",
        "localhost",
        "github.com",
        "gw.example",
        "shell.example",
        "settings.example",
        "from-toml.example",
        "from-env.example",
        "from-flag.example",
        "from-client.example",
        "gateway.example.com",
    }
    url_re = re.compile(r"https?://([A-Za-z0-9._-]+)")
    offenders: list[str] = []

    scanned = (
        list(REPO.glob("lingua_proxy/**/*.py"))
        + list(REPO.glob("lingua_proxy/**/*.jsonl"))
        + list(REPO.glob("tests/**/*.py"))
        + list(REPO.glob("tests/fixtures/*.json"))
        + [REPO / "pyproject.toml"]
    )
    for path in scanned:
        if not path.exists():
            continue
        for host in url_re.findall(path.read_text()):
            bare = host.split(":")[0]
            # RFC 6761 reserves .test/.example for documentation and testing;
            # neither can resolve to a real private gateway.
            if (
                bare in allowed_hosts
                or bare.endswith(".test")
                or bare.endswith(".example")
                or bare.endswith(".example.com")
            ):
                continue
            offenders.append(f"{path.relative_to(REPO)}: {host}")

    assert not offenders, f"private or unexpected endpoints committed: {offenders}"
