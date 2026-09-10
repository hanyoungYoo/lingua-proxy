"""Environment checks.

Each check answers one question a confused user would ask, and every failure
carries a hint that says what to do next. Credentials are checked for presence
and never printed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import httpx

from lingua_proxy.config import (
    default_client_settings_path,
    lingua_home,
    resolve_anthropic_upstream,
)

PASS, WARN, FAIL = "pass", "warn", "fail"
_SEVERITY = {PASS: 0, WARN: 1, FAIL: 2}
_GLYPH = {PASS: "[green]✓[/green]", WARN: "[yellow]⚠[/yellow]", FAIL: "[red]✗[/red]"}


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    hint: str = ""

    @property
    def severity(self) -> int:
        return _SEVERITY[self.status]

    @property
    def glyph(self) -> str:
        return _GLYPH[self.status]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "hint": self.hint,
        }


def check_python() -> CheckResult:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    # Packaging metadata already requires 3.12, but someone running from a
    # source checkout on an older interpreter deserves a clear message rather
    # than an import error deep in a dependency.
    if sys.version_info < (3, 12):  # noqa: UP036
        return CheckResult(
            "Python version", FAIL, version, "lingua-proxy needs Python 3.12 or newer."
        )
    return CheckResult("Python version", PASS, version)


def check_platform() -> CheckResult:
    if os.name != "posix":
        return CheckResult(
            "Platform", WARN, os.name, "Windows is not supported yet; some features may fail."
        )
    return CheckResult("Platform", PASS, sys.platform)


def check_detector() -> CheckResult:
    try:
        from lingua_proxy.detector import Detector

        Detector().detect("이 문장은 한국어로 작성되었습니다.")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Language detection", FAIL, str(exc), "Reinstall lingua-language-detector."
        )
    return CheckResult("Language detection", PASS, "models load and detect")


def check_proxy(port: int) -> CheckResult:
    try:
        response = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
        body = response.json()
    except Exception:
        return CheckResult(
            "Proxy",
            FAIL,
            f"nothing answering on port {port}",
            f"Start it with 'lingua-proxy proxy --port {port}'.",
        )

    if body.get("service") != "lingua-proxy":
        return CheckResult(
            "Proxy",
            FAIL,
            f"port {port} is serving something else",
            "Choose a free port with --port.",
        )
    return CheckResult("Proxy", PASS, f"running on port {port} (v{body.get('version')})")


def check_upstream() -> CheckResult:
    url = resolve_anthropic_upstream()
    try:
        response = httpx.get(url.rstrip("/") + "/v1/models", timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Upstream",
            FAIL,
            f"{url} unreachable ({type(exc).__name__})",
            "Check the URL and your network, or pass --upstream.",
        )

    # 401/403 means we reached a real API that wants credentials, which is fine.
    if response.status_code in (200, 401, 403):
        return CheckResult("Upstream", PASS, f"{url} reachable")
    return CheckResult(
        "Upstream",
        WARN,
        f"{url} returned HTTP {response.status_code}",
        "The upstream answered unexpectedly; verify the base URL.",
    )


def check_credentials() -> CheckResult:
    """Confirm a credential exists. Never print its value."""
    if os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"):
        return CheckResult("Credentials", PASS, "found in the environment")

    path = default_client_settings_path()
    try:
        env = json.loads(path.read_text()).get("env", {})
    except (OSError, ValueError, AttributeError):
        env = {}

    if env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY"):
        return CheckResult("Credentials", PASS, "found in client settings")

    return CheckResult(
        "Credentials",
        WARN,
        "none found",
        "The proxy forwards whatever your client sends, so this is only a problem "
        "if your client has no credential either.",
    )


def check_storage() -> CheckResult:
    home = lingua_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write-probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return CheckResult("Local storage", FAIL, str(exc), f"Make {home} writable.")
    return CheckResult("Local storage", PASS, str(home))


def run_checks(port: int = 8787) -> list[CheckResult]:
    return [
        check_python(),
        check_platform(),
        check_detector(),
        check_storage(),
        check_proxy(port),
        check_upstream(),
        check_credentials(),
    ]
