"""FastAPI application: routing, upstream forwarding and byte-exact relay.

v0.1 responsibilities of this module are deliberately narrow -- it is a
faithful reverse proxy. Translation is layered on top by ``pipeline.py``;
anything this module cannot handle falls through to a transparent relay.

Fidelity rules that matter and are easy to get wrong:

* ``accept-encoding: identity`` is forced upstream. Otherwise httpx negotiates
  gzip, transparently decodes, and a "byte-exact" relay silently is not.
* Upstream error bodies are relayed untouched, because clients (Claude Code in
  particular) match on the upstream's own error wording to decide whether to
  retry or disable a capability.
* Hop-by-hop headers are stripped from the response; everything else, including
  rate-limit and request-id headers, is passed on.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from lingua_proxy import __version__
from lingua_proxy.codecs import AnthropicMessagesCodec, Codec, OpenAIChatCodec
from lingua_proxy.config import Settings, join_upstream
from lingua_proxy.cost_log import CostLog, CostRow, estimate_tokens, summarize
from lingua_proxy.detector import Detector
from lingua_proxy.memo import Memo
from lingua_proxy.pipeline import Pipeline, TranslationSkipped, parse_body
from lingua_proxy.streaming import (
    iter_sse_events,
    translate_anthropic_stream,
    translate_openai_stream,
)
from lingua_proxy.translator import LLMTranslator

# Headers we must not copy from the inbound request to the upstream.
_DROP_REQUEST_HEADERS = frozenset({"host", "content-length", "accept-encoding", "connection"})

# Hop-by-hop headers that must not be copied from the upstream to the client.
_DROP_RESPONSE_HEADERS = frozenset(
    {"transfer-encoding", "connection", "keep-alive", "content-encoding", "content-length"}
)

# Our own control headers never reach the upstream.
_INTERNAL_PREFIX = "x-lingua-"


def build_upstream_headers(request: Request) -> dict[str, str]:
    """Copy inbound headers for the upstream call.

    Everything is forwarded except hop-by-hop headers and our own controls.
    Notably ``authorization``, ``x-api-key``, ``anthropic-version`` and
    ``anthropic-beta`` pass through verbatim -- ``anthropic-beta`` is an open,
    fast-moving list and allowlisting it breaks on the next release.
    """
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _DROP_REQUEST_HEADERS and not key.lower().startswith(_INTERNAL_PREFIX)
    }
    headers["accept-encoding"] = "identity"
    return headers


def build_response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in _DROP_RESPONSE_HEADERS
    }


async def stream_upstream(upstream: httpx.Response) -> AsyncIterator[bytes]:
    """Relay raw upstream bytes, closing the upstream when the client leaves."""
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    finally:
        await upstream.aclose()


def upstream_error_response(exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "type": "error",
            "error": {
                "type": "lingua_proxy_upstream_error",
                "message": f"lingua-proxy could not reach the upstream: {exc}",
            },
        },
    )


def _wants_stream(body: bytes) -> bool:
    """Cheap check for ``"stream": true`` without paying for a full parse."""
    return b'"stream"' in body and b"true" in body


async def relay(request: Request, upstream_base: str) -> Response:
    """Forward one request upstream and relay the reply verbatim."""
    client: httpx.AsyncClient = request.app.state.client
    body = await request.body()
    url = join_upstream(upstream_base, request.url.path, request.url.query)
    headers = build_upstream_headers(request)

    upstream_request = client.build_request(
        request.method, url, headers=headers, content=body or None
    )

    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return upstream_error_response(exc)

    content_type = upstream.headers.get("content-type", "")
    is_sse = "text/event-stream" in content_type

    # Errors are always buffered so the status and body reach the client as one
    # piece, exactly as the upstream wrote them.
    if upstream.status_code >= 400 or not is_sse:
        try:
            payload = await upstream.aread()
        except httpx.HTTPError as exc:
            await upstream.aclose()
            return upstream_error_response(exc)
        await upstream.aclose()
        return Response(
            content=payload,
            status_code=upstream.status_code,
            headers=build_response_headers(upstream),
            media_type=content_type or None,
        )

    return StreamingResponse(
        stream_upstream(upstream),
        status_code=upstream.status_code,
        headers=build_response_headers(upstream),
        media_type="text/event-stream",
    )


_BYPASS_VALUES = frozenset({"true", "1", "yes"})


def _wants_bypass(request: Request) -> bool:
    return request.headers.get("x-lingua-bypass", "").strip().lower() in _BYPASS_VALUES


def _get_pipeline(request: Request) -> Pipeline:
    """Build the pipeline for this request.

    The detector and memo are process-wide, but the translator carries the
    caller's own credentials, so it is constructed per request unless a
    translator was injected for testing.
    """
    app = request.app
    settings: Settings = app.state.settings

    if app.state.detector is None:
        # Loading language models is slow; do it once, on first use.
        app.state.detector = app.state.detector_factory()

    translator = app.state.translator
    if translator is None:
        translator = app.state.translator_factory(_translator_auth(request))

    return Pipeline(
        detector=app.state.detector,
        translator=translator,
        memo=app.state.memo,
        skip_models=settings.skip_models,
    )


def _translator_auth(request: Request) -> dict[str, str]:
    """Credentials the translator reuses, copied from the caller."""
    wanted = ("authorization", "x-api-key", "anthropic-version", "anthropic-beta", "user-agent")
    return {key: request.headers[key] for key in wanted if key in request.headers}


async def forward_json(
    request: Request, upstream_base: str, body: bytes
) -> httpx.Response | Response:
    """POST a (possibly rewritten) JSON body upstream and read the reply."""
    client: httpx.AsyncClient = request.app.state.client
    url = join_upstream(upstream_base, request.url.path, request.url.query)
    headers = build_upstream_headers(request)
    headers["content-length"] = str(len(body))

    try:
        upstream = await client.post(url, headers=headers, content=body)
    except httpx.HTTPError as exc:
        return upstream_error_response(exc)
    return upstream


def record_cost(
    request: Request,
    codec: Codec,
    original_body: dict,
    forwarded_body: dict,
    english_response: dict,
    localized_response: dict,
    outcome,
    started: float,
) -> None:
    """Write one accounting row.

    The counterfactual is measured as a *delta*: how many more tokens the
    original wording would have cost than the English we actually sent. Doing
    it that way cancels framing overhead and reasoning tokens, which are
    identical either way and would otherwise inflate the reported saving.
    """
    settings: Settings = request.app.state.settings
    usage = codec.usage(english_response)

    original_texts = [ref.text for ref in codec.user_refs(original_body)]
    english_texts = [ref.text for ref in codec.user_refs(forwarded_body)]
    input_delta = sum(estimate_tokens(t) for t in original_texts) - sum(
        estimate_tokens(t) for t in english_texts
    )

    reply_english = [ref.text for ref in codec.response_refs(english_response)]
    reply_localized = [ref.text for ref in codec.response_refs(localized_response)]
    output_delta = sum(estimate_tokens(t) for t in reply_localized) - sum(
        estimate_tokens(t) for t in reply_english
    )

    row = CostRow(
        model=codec.model(original_body),
        translator_model=settings.translator_model,
        source_lang=outcome.source_lang,
        endpoint=request.url.path,
        translated=outcome.translated,
        usage=usage,
        counterfactual_input=max(0, usage.total_input + input_delta),
        counterfactual_output=max(0, usage.output_tokens + output_delta),
        estimate_method="heuristic",
        translator_calls=outcome.translator_calls,
        memo_hits=outcome.memo_hits,
        memo_miss_assistant=outcome.memo_miss_assistant,
        fallback_reason=outcome.fallback_reason,
        latency_ms=(time.monotonic() - started) * 1000,
    )
    with contextlib.suppress(OSError):
        request.app.state.cost_log.record(row)


async def stream_translated(
    request: Request,
    codec: Codec,
    upstream_base: str,
    payload: bytes,
    pipeline: Pipeline,
    source: str,
    outcome,
) -> Response:
    """Forward a streaming request and translate its text blocks in flight."""
    client: httpx.AsyncClient = request.app.state.client
    settings: Settings = request.app.state.settings
    url = join_upstream(upstream_base, request.url.path, request.url.query)
    headers = build_upstream_headers(request)
    headers["content-length"] = str(len(payload))

    upstream_request = client.build_request("POST", url, headers=headers, content=payload)
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return upstream_error_response(exc)

    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=body,
            status_code=upstream.status_code,
            headers=build_response_headers(upstream),
            media_type=upstream.headers.get("content-type") or None,
        )

    async def translate(texts: list[str]) -> list[str]:
        """Translate streamed text, preserving order and never losing a block.

        A missing translation falls back to the original English: an answer
        the user has to read in English still beats no answer at all.
        """
        produced = await pipeline.translate_texts_for_stream(texts, source, outcome)
        return [produced.get(text, text) for text in texts]

    is_anthropic = isinstance(codec, AnthropicMessagesCodec)
    relay_fn = translate_anthropic_stream if is_anthropic else translate_openai_stream

    async def body_iter():
        try:
            async for chunk in relay_fn(
                iter_sse_events(upstream.aiter_raw()),
                translate,
                ping_interval=settings.ping_interval,
            ):
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=build_response_headers(upstream),
        media_type="text/event-stream",
    )


async def handle(request: Request, codec: Codec, upstream_base: str) -> Response:
    """Translate, forward, translate back.

    Anything that is not a plain JSON request we understand, or any decision
    not to translate, falls through to the byte-exact relay.
    """
    started = time.monotonic()
    raw = await request.body()
    body = parse_body(raw)
    if body is None:
        return await relay(request, upstream_base)

    pipeline = _get_pipeline(request)

    if pipeline.should_skip(body, codec, _wants_bypass(request)) is not None:
        return await relay(request, upstream_base)

    source = pipeline.conversation_language(body, codec)
    if source is None:
        return await relay(request, upstream_base)

    try:
        forwarded, outcome = await pipeline.translate_request(body, codec, source)
    except TranslationSkipped:
        return await relay(request, upstream_base)

    payload = json.dumps(forwarded, ensure_ascii=False).encode()

    if codec.is_stream(body):
        return await stream_translated(
            request, codec, upstream_base, payload, pipeline, source, outcome
        )

    upstream = await forward_json(request, upstream_base, payload)
    if isinstance(upstream, Response):
        return upstream

    if upstream.status_code >= 400:
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=build_response_headers(upstream),
            media_type=upstream.headers.get("content-type") or None,
        )

    try:
        response_body = upstream.json()
    except ValueError:
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=build_response_headers(upstream),
            media_type=upstream.headers.get("content-type") or None,
        )

    localized = await pipeline.translate_response(response_body, codec, source, outcome)

    record_cost(request, codec, body, forwarded, response_body, localized, outcome, started)

    return JSONResponse(
        content=localized,
        status_code=upstream.status_code,
        headers=build_response_headers(upstream),
    )


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    translator=None,
    detector: Detector | None = None,
) -> FastAPI:
    """Build the ASGI app.

    ``transport``, ``translator`` and ``detector`` are seams for tests:
    passing them replaces the upstream, the translation model and the
    (slow to build) language detector without touching the network.
    """
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="lingua-proxy", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.anthropic_codec = AnthropicMessagesCodec()
    app.state.openai_codec = OpenAIChatCodec()
    app.state.memo = Memo(
        path=settings.memo_path,
        persist=settings.memo_persist,
        max_entries=settings.memo_max_entries,
    )
    app.state.pipeline = None
    app.state.detector = detector
    app.state.cost_log = CostLog(path=settings.cost_log_path)
    # A real translator is built on first use when one was not injected.
    # Without this the pipeline would fail open on every request and silently
    # behave as a plain passthrough.
    app.state.translator = translator
    app.state.translator_factory = lambda auth_headers: LLMTranslator(
        upstream_url=settings.translator_base_url or settings.upstream_anthropic_url,
        client=app.state.client,
        model=settings.translator_model,
        auth_headers=auth_headers,
        api_key=settings.translator_api_key,
    )
    app.state.detector_factory = lambda: Detector(
        languages=settings.languages,
        min_chars=settings.min_chars,
        min_confidence=settings.min_confidence,
    )
    # Created eagerly rather than in the lifespan so the app works under
    # ASGITransport, which never runs startup events.
    app.state.client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(600.0, connect=10.0),
        follow_redirects=False,
    )

    @app.get("/stats")
    async def stats() -> dict:
        return summarize(app.state.cost_log.rows()).to_json()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"service": "lingua-proxy", "version": __version__}

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        return await handle(request, app.state.anthropic_codec, settings.upstream_anthropic_url)

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def openai_chat(request: Request) -> Response:
        return await handle(request, app.state.openai_codec, settings.upstream_openai_url)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def passthrough(request: Request, path: str) -> Response:
        return await relay(request, settings.upstream_anthropic_url)

    return app
