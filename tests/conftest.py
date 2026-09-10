"""Shared test fixtures.

Two rules drive this design:

1. No test may touch a real network. Upstreams are served by
   ``RecordingTransport``, a ``httpx.MockTransport`` that also records every
   request so tests can assert on exactly what was forwarded.
2. Translation must be deterministic. ``FakeTranslator`` marks text with a
   language prefix instead of calling a model, which is invertible and leaves
   masked placeholders intact.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

# --------------------------------------------------------------------------
# canned upstream responses
# --------------------------------------------------------------------------


@dataclass
class Canned:
    """One scripted upstream reply.

    ``gate`` lets a test hold the response open (or hold back individual
    stream chunks) to prove ordering and keep-alive behaviour.
    """

    status: int = 200
    json_body: Any | None = None
    text: str | None = None
    chunks: list[bytes] | None = None
    headers: dict[str, str] = field(default_factory=dict)
    gate: asyncio.Event | None = None
    chunk_gates: list[asyncio.Event] | None = None
    exc: Exception | None = None

    def body_bytes(self) -> bytes:
        if self.json_body is not None:
            return json.dumps(self.json_body).encode()
        if self.text is not None:
            return self.text.encode()
        return b""


def sse(*events: tuple[str, dict[str, Any]]) -> list[bytes]:
    """Build a list of SSE frames, one bytes chunk per event."""
    out = []
    for name, data in events:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        out.append(f"event: {name}\ndata: {payload}\n\n".encode())
    return out


class RecordingTransport(httpx.AsyncBaseTransport):
    """Mock upstream that records requests and replays a script of responses.

    ``script`` may be a list (consumed in order, the last entry repeats) or a
    callable taking the request and returning a ``Canned``.
    """

    def __init__(self, script: list[Canned] | Callable[[httpx.Request], Canned] | Canned):
        if isinstance(script, Canned):
            script = [script]
        self.script = script
        self.requests: list[httpx.Request] = []
        self.bodies: list[bytes] = []
        self._i = 0

    def _next(self, request: httpx.Request) -> Canned:
        if callable(self.script):
            return self.script(request)
        if not self.script:
            return Canned(status=200, json_body={})
        idx = min(self._i, len(self.script) - 1)
        self._i += 1
        return self.script[idx]

    @property
    def json_bodies(self) -> list[Any]:
        return [json.loads(b) for b in self.bodies if b]

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        self.requests.append(request)
        self.bodies.append(body)
        canned = self._next(request)

        if canned.exc is not None:
            raise canned.exc
        if canned.gate is not None:
            await canned.gate.wait()

        headers = dict(canned.headers)

        if canned.chunks is not None:
            headers.setdefault("content-type", "text/event-stream")
            gates = canned.chunk_gates

            async def stream() -> Iterable[bytes]:
                for i, chunk in enumerate(canned.chunks or []):
                    if gates and i < len(gates) and gates[i] is not None:
                        await gates[i].wait()
                    yield chunk

            return httpx.Response(canned.status, headers=headers, content=stream())

        payload = canned.body_bytes()
        if canned.json_body is not None:
            headers.setdefault("content-type", "application/json")
        return httpx.Response(canned.status, headers=headers, content=payload)


# --------------------------------------------------------------------------
# deterministic translator
# --------------------------------------------------------------------------


class FakeTranslator:
    """Deterministic stand-in for a translation model.

    Returns ``«{target}»{text}`` unless the exact text is in ``mapping``. The
    prefix marker is invertible and, unlike reversing the string, leaves
    ``<lpN/>`` placeholders untouched.
    """

    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = dict(mapping or {})
        self.calls: list[tuple[list[str], str, str]] = []
        self.fail_with: Exception | None = None

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def translated_segments(self) -> list[str]:
        return [seg for call in self.calls for seg in call[0]]

    async def translate(self, segments: list[str], source: str, target: str) -> list[str]:
        self.calls.append((list(segments), source, target))
        if self.fail_with is not None:
            raise self.fail_with
        return [self.mapping.get(s, f"«{target}»{s}") for s in segments]


@pytest.fixture
def fake_translator() -> FakeTranslator:
    return FakeTranslator()


# --------------------------------------------------------------------------
# filesystem isolation
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never read or write the developer's real ~/.lingua-proxy or ~/.claude."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LINGUA_HOME", str(home / ".lingua-proxy"))
    for var in (
        "LINGUA_UPSTREAM_ANTHROPIC_URL",
        "LINGUA_UPSTREAM_OPENAI_URL",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "LINGUA_BENCH",
        "LINGUA_BENCH_BASE_URL",
        "LINGUA_BENCH_AUTH",
        "LINGUA_PROXY_WRAPPED",
    ):
        monkeypatch.delenv(var, raising=False)
    return home
