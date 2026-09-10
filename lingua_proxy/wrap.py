"""Transactional client patching.

``wrap`` edits a settings file that belongs to another program, so it behaves
like a transaction: record the previous value, patch, run, restore on every
exit path including signals. A marker file kept outside the project lets a
later ``unwrap`` clean up after a crash without leaving anything in the repo.

It never invents an upstream. Whatever base URL the client is already using
becomes lingua-proxy's upstream, so a user behind a company gateway keeps
their gateway.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from dataclasses import dataclass

from lingua_proxy.config import lingua_home

SETTINGS_KEY = "ANTHROPIC_BASE_URL"


class WrapError(RuntimeError):
    """Raised when wrapping cannot proceed safely."""


def project_settings_path(cwd: pathlib.Path | None = None) -> pathlib.Path:
    root = cwd or pathlib.Path.cwd()
    return root / ".claude" / "settings.local.json"


def marker_path(cwd: pathlib.Path | None = None) -> pathlib.Path:
    """Marker lives in our own directory, never inside the user's repo."""
    root = str((cwd or pathlib.Path.cwd()).resolve())
    digest = hashlib.sha1(root.encode()).hexdigest()[:16]
    return lingua_home() / "wraps" / f"{digest}.json"


@dataclass
class WrapState:
    """What we changed, and how to put it back."""

    settings_path: str
    previous: str | None
    created_file: bool
    created_env: bool
    port: int

    def to_json(self) -> dict:
        return {
            "settings_path": self.settings_path,
            "previous": self.previous,
            "created_file": self.created_file,
            "created_env": self.created_env,
            "port": self.port,
        }

    @classmethod
    def from_json(cls, data: dict) -> WrapState:
        return cls(
            settings_path=data["settings_path"],
            previous=data.get("previous"),
            created_file=bool(data.get("created_file")),
            created_env=bool(data.get("created_env")),
            port=int(data.get("port") or 0),
        )


def _read_settings(path: pathlib.Path) -> dict:
    """Read settings, refusing to proceed on a file we cannot parse.

    Overwriting a malformed file would destroy configuration we do not own.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WrapError(
            f"{path} is not valid JSON, so lingua-proxy will not modify it. "
            "Fix or move the file and try again."
        ) from exc
    if not isinstance(data, dict):
        raise WrapError(f"{path} does not contain a JSON object.")
    return data


def _write_settings(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def apply_wrap(proxy_url: str, port: int, *, cwd: pathlib.Path | None = None) -> WrapState:
    """Point the client at the proxy, remembering how to undo it."""
    path = project_settings_path(cwd)
    created_file = not path.exists()
    payload = _read_settings(path)

    env = payload.get("env")
    created_env = not isinstance(env, dict)
    if created_env:
        env = {}

    state = WrapState(
        settings_path=str(path),
        previous=env.get(SETTINGS_KEY),
        created_file=created_file,
        created_env=created_env,
        port=port,
    )

    env[SETTINGS_KEY] = proxy_url
    payload["env"] = env
    _write_settings(path, payload)

    marker = marker_path(cwd)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(state.to_json(), indent=2))
    return state


def restore_wrap(state: WrapState) -> None:
    """Undo :func:`apply_wrap`, leaving no trace behind."""
    path = pathlib.Path(state.settings_path)
    if not path.exists():
        return

    try:
        payload = _read_settings(path)
    except WrapError:
        # The user edited it into an invalid state while we ran; leave it be.
        return

    env = payload.get("env")
    if isinstance(env, dict):
        if state.previous is not None:
            env[SETTINGS_KEY] = state.previous
        else:
            env.pop(SETTINGS_KEY, None)
        if state.created_env and not env:
            payload.pop("env", None)
        else:
            payload["env"] = env

    if state.created_file and not payload:
        path.unlink(missing_ok=True)
        parent = path.parent
        # Remove the .claude directory only if we created it and it is empty.
        try:
            next(parent.iterdir())
        except StopIteration:
            parent.rmdir()
        except OSError:
            pass
    else:
        _write_settings(path, payload)


def clear_marker(cwd: pathlib.Path | None = None) -> None:
    marker_path(cwd).unlink(missing_ok=True)


def load_marker(cwd: pathlib.Path | None = None) -> WrapState | None:
    marker = marker_path(cwd)
    if not marker.exists():
        return None
    try:
        return WrapState.from_json(json.loads(marker.read_text()))
    except (ValueError, KeyError):
        return None


def find_claude_binary() -> str:
    """Locate the Claude Code CLI, including its common install locations."""
    found = shutil.which("claude")
    if found:
        return found

    for candidate in (
        pathlib.Path.home() / ".local" / "bin" / "claude",
        pathlib.Path.home() / ".claude" / "local" / "claude",
    ):
        if candidate.exists():
            return str(candidate)

    raise WrapError(
        "Could not find the 'claude' command on your PATH. "
        "Install Claude Code, or run 'lingua-proxy proxy' and point your client "
        "at it yourself."
    )
