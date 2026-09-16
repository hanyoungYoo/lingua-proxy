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


#: Settings a user may set in ``config.toml``, with the type each must have.
#: Upstream URLs are resolved separately because they also consult the
#: environment and the client's own settings file.
_TOML_SETTINGS: dict[str, type | tuple[type, ...]] = {
    "host": str,
    "port": int,
    "translator_model": str,
    "translator_base_url": str,
    "translator_api_key": str,
    "translator_format": str,
    "min_chars": int,
    "min_confidence": (int, float),
    "memo_persist": bool,
    "memo_max_entries": int,
    "count_tokens_enabled": bool,
    "ping_interval": (int, float),
    "max_output_tokens_for_translation": int,
    "preserve_formatting": bool,
    "reply_only_prose_share": (int, float),
}

#: Settings that name a filesystem path.
_TOML_PATH_SETTINGS = ("memo_path", "cost_log_path", "audit_log_path")

#: Settings that are a list of strings.
_TOML_TUPLE_SETTINGS = ("skip_models", "languages")


def _settings_from_toml(config_path: pathlib.Path | None = None) -> dict:
    """Read recognised settings out of the config file.

    Unknown keys and wrongly typed values are ignored rather than fatal. The
    file is hand-edited, and a typo in one option should not stop the proxy
    from starting with sane defaults for the rest.
    """
    cfg = _read_toml(config_path if config_path is not None else default_config_path())
    if not cfg:
        return {}

    out: dict = {}
    for key, expected in _TOML_SETTINGS.items():
        if key not in cfg:
            continue
        value = cfg[key]
        # bool is a subclass of int, so an explicit check keeps `port = true`
        # from being accepted as an integer.
        if expected is not bool and isinstance(value, bool):
            continue
        if isinstance(value, expected):
            out[key] = float(value) if expected == (int, float) else value

    for key in _TOML_PATH_SETTINGS:
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = pathlib.Path(value).expanduser()

    for key in _TOML_TUPLE_SETTINGS:
        value = cfg.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            out[key] = tuple(value)

    return out


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

    # Whether translation pays depends on how much shorter the English answer
    # is, not on how long it is: both the saving and the fee scale with the
    # answer, so the length cancels out. On Sonnet with a Haiku translator the
    # English reply must be ~38% shorter to break even.
    #
    # A very large max_tokens is nonetheless a useful warning sign, because a
    # reply that hits the ceiling is pinned to the same length in both
    # languages, which guarantees zero shrink and a fee charged for nothing.
    # Set to 0 to disable.
    max_output_tokens_for_translation: int = 16000

    # Models format differently by language: the same question answered in
    # Korean comes back with a heading and a numbered list, answered in English
    # it comes back as prose. Translating that English faithfully still loses
    # the structure, so the model is asked to keep its usual formatting.
    # On by default: silently flattening an answer is not something a user
    # opted into. Disable globally here, or per request with the
    # x-lingua-preserve-formatting header.
    preserve_formatting: bool = True

    # A prompt that is mostly code masks down to almost nothing before it
    # reaches the translator, so translating the request costs a call and saves
    # close to zero. Skipping translation outright would also skip the saving,
    # though: the model would answer the Korean prompt in Korean, and the
    # expensive model writing Korean is the cost this proxy exists to avoid.
    #
    # So below this prose share the request is forwarded in the user's own
    # language with an instruction to answer in English, and only the reply is
    # translated. The saving comes from the reply either way; this drops the
    # request-side fee that was buying nothing. Set to 0.0 to disable.
    reply_only_prose_share: float = 0.35

    @classmethod
    def resolve(
        cls,
        *,
        upstream: str | None = None,
        openai_upstream: str | None = None,
        config_path: pathlib.Path | None = None,
        **overrides,
    ) -> Settings:
        """Build settings from the config file, then apply explicit overrides.

        Values passed by the caller win over the config file, which wins over
        the defaults.
        """
        from_file = _settings_from_toml(config_path)
        from_file.update(overrides)
        return cls(
            upstream_anthropic_url=resolve_anthropic_upstream(upstream, config_path=config_path),
            upstream_openai_url=resolve_openai_upstream(openai_upstream, config_path=config_path),
            **from_file,
        )
