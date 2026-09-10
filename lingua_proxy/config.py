"""Configuration and upstream resolution.

Design rule: **no endpoint is ever hard-coded except the public vendor APIs.**
Many users already point their client at a company gateway, LiteLLM, a
self-hosted relay, or another local proxy. lingua-proxy chains onto whatever is
already configured, preserving scheme, host, port and path prefix.

Resolution order, highest first:

1. explicit CLI flag
2. ``LINGUA_UPSTREAM_ANTHROPIC_URL`` / ``LINGUA_UPSTREAM_OPENAI_URL``
3. ``~/.lingua-proxy/config.toml``
4. the base URL the client already uses (``~/.claude/settings.json`` env block,
   then the shell ``ANTHROPIC_BASE_URL``)
5. the public API
"""

from __future__ import annotations

import json
import os
import pathlib
import tomllib
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

DEFAULT_ANTHROPIC_API_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_API_URL = "https://api.openai.com"

# Small helper models that agentic clients call in the background. Translating
# them costs money and buys nothing the user ever reads.
DEFAULT_SKIP_MODELS = ("claude-haiku-*", "*haiku*")

# Languages worth translating by default: English plus non-Latin scripts, where
# the tokenization penalty is large. Latin-script languages are opt-in because
# the saving is small and short-English false positives are common.
DEFAULT_LANGUAGES = ("en", "ko", "ja", "zh", "ar", "ru", "hi", "th")


def lingua_home() -> pathlib.Path:
    """Directory for local state. Honours ``LINGUA_HOME`` for tests."""
    env = os.environ.get("LINGUA_HOME")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(os.path.expanduser("~")) / ".lingua-proxy"


def default_config_path() -> pathlib.Path:
    return lingua_home() / "config.toml"


def default_client_settings_path() -> pathlib.Path:
    return pathlib.Path(os.path.expanduser("~")) / ".claude" / "settings.json"


def join_upstream(base: str, path: str, query: str = "") -> str:
    """Join an upstream base with a request path, preserving any path prefix.

    ``https://gw.example/anthropic/`` + ``/v1/messages`` becomes
    ``https://gw.example/anthropic/v1/messages`` -- the prefix survives, which
    a naive ``urljoin`` would discard.
    """
    parts = urlsplit(base)
    prefix = parts.path.rstrip("/")
    joined = prefix + path if path.startswith("/") else f"{prefix}/{path}"
    return urlunsplit((parts.scheme, parts.netloc, joined, query, ""))


def _read_toml(path: pathlib.Path) -> dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _client_settings_base_url(path: pathlib.Path) -> str | None:
    """Read ``env.ANTHROPIC_BASE_URL`` from a client settings file.

    A malformed settings file is ignored rather than fatal: it belongs to
    another tool and we are only reading it opportunistically.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    env = data.get("env")
    if not isinstance(env, dict):
        return None
    value = env.get("ANTHROPIC_BASE_URL")
    return value if isinstance(value, str) and value.strip() else None


def resolve_anthropic_upstream(
    flag: str | None = None,
    *,
    config_path: pathlib.Path | None = None,
    client_settings_path: pathlib.Path | None = None,
) -> str:
    if flag:
        return flag
    env = os.environ.get("LINGUA_UPSTREAM_ANTHROPIC_URL")
    if env:
        return env
    cfg = _read_toml(config_path if config_path is not None else default_config_path())
    if isinstance(cfg.get("upstream_anthropic_url"), str):
        return cfg["upstream_anthropic_url"]

    settings_path = (
        client_settings_path if client_settings_path is not None else default_client_settings_path()
    )
    from_client = _client_settings_base_url(settings_path)
    if from_client:
        return from_client

    shell = os.environ.get("ANTHROPIC_BASE_URL")
    if shell:
        return shell
    return DEFAULT_ANTHROPIC_API_URL


def resolve_openai_upstream(
    flag: str | None = None, *, config_path: pathlib.Path | None = None
) -> str:
    if flag:
        return flag
    env = os.environ.get("LINGUA_UPSTREAM_OPENAI_URL")
    if env:
        return env
    cfg = _read_toml(config_path if config_path is not None else default_config_path())
    if isinstance(cfg.get("upstream_openai_url"), str):
        return cfg["upstream_openai_url"]
    return DEFAULT_OPENAI_API_URL


@dataclass
class Settings:
    """Runtime settings for one proxy process."""

    host: str = "127.0.0.1"
    port: int = 8787
    upstream_anthropic_url: str = DEFAULT_ANTHROPIC_API_URL
    upstream_openai_url: str = DEFAULT_OPENAI_API_URL

    # translation
    translator_model: str = "claude-haiku-4-5"
    translator_base_url: str | None = None
    translator_api_key: str | None = None
    translator_format: str = "anthropic"
    skip_models: tuple[str, ...] = DEFAULT_SKIP_MODELS

    # detection
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    min_chars: int = 20
    min_confidence: float = 0.6

    # storage
    memo_path: pathlib.Path = field(default_factory=lambda: lingua_home() / "memo.jsonl")
    memo_persist: bool = True
    memo_max_entries: int = 50_000
    cost_log_path: pathlib.Path = field(default_factory=lambda: lingua_home() / "cost_log.jsonl")
    count_tokens_enabled: bool = True

    # streaming
    ping_interval: float = 15.0

    # Transparency. A wrong translation is fluent and passes every structural
    # check, so the defence is visibility rather than detection: the audit log
    # records both sides of every rewrite. Off by default because it writes
    # prompt text to disk.
    audit_log_path: pathlib.Path | None = None

    # Translation cost scales with the length of the reply, while the saving
    # does not. Past a few thousand output tokens the fee outruns the benefit,
    # so a request that asks for a very long answer is passed through instead.
    # Measured: a bounded answer saves ~49%, an open-ended one costs ~40% more.
    # Set to 0 to disable the guard.
    max_output_tokens_for_translation: int = 4000

    @classmethod
    def resolve(
        cls,
        *,
        upstream: str | None = None,
        openai_upstream: str | None = None,
        **overrides,
    ) -> Settings:
        return cls(
            upstream_anthropic_url=resolve_anthropic_upstream(upstream),
            upstream_openai_url=resolve_openai_upstream(openai_upstream),
            **overrides,
        )
