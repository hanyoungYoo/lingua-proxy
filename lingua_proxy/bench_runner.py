"""Live benchmark execution.

Each corpus prompt is sent twice through the proxy: once normally, and once
with ``x-lingua-bypass`` so the model answers in the original language. That
gives a measured baseline instead of an estimated one, which matters because
an estimate is exactly the thing a sceptical reader would not trust.

Every request costs money, so this is opt-in and never runs by default.
"""

from __future__ import annotations

import os
import pathlib
import time

import httpx

from lingua_proxy.bench import BenchResult, build_report, load_corpus
from lingua_proxy.codecs import AnthropicMessagesCodec
from lingua_proxy.config import Settings
from lingua_proxy.cost_log import CostLog, price_for
from lingua_proxy.proxy import create_app

DEFAULT_MODEL = "claude-sonnet-4-6"
# Generous enough that ordinary replies finish naturally. If a reply is cut
# off at the ceiling, both runs produce identical output token counts and the
# measurement understates savings, because output is where most cost lives.
MAX_TOKENS = 4096


def _auth_headers() -> dict[str, str]:
    headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    token = os.environ.get("LINGUA_BENCH_AUTH")
    if token:
        if token.startswith("sk-"):
            headers["x-api-key"] = token
        else:
            headers["authorization"] = f"Bearer {token}"
    return headers


async def _one(
    client: httpx.AsyncClient,
    row: dict,
    model: str,
    codec: AnthropicMessagesCodec,
) -> BenchResult:
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": row["prompt"]}],
    }
    headers = _auth_headers()
    price = price_for(model)

    started = time.monotonic()
    try:
        # Baseline: the model answers in the original language.
        native = await client.post(
            "/v1/messages", json=body, headers={**headers, "x-lingua-bypass": "true"}, timeout=120
        )
        native.raise_for_status()
        baseline_usage = codec.usage(native.json())

        # Proxied: translated in and back out.
        proxied = await client.post("/v1/messages", json=body, headers=headers, timeout=180)
        proxied.raise_for_status()
        proxied_usage = codec.usage(proxied.json())
    except Exception as exc:  # noqa: BLE001 - one bad prompt must not end the run
        return BenchResult(row["id"], row["lang"], row["category"], error=str(exc)[:200])

    translator_cost = 0.0
    log_path = getattr(client, "_lingua_cost_log", None)
    if log_path is not None:
        rows = CostLog(path=log_path).rows()
        if rows:
            last = rows[-1]
            translator_cost = price_for(last.translator_model).cost(last.translator_usage)

    return BenchResult(
        id=row["id"],
        lang=row["lang"],
        category=row["category"],
        baseline_cost=price.cost(baseline_usage),
        proxied_cost=price.cost(proxied_usage) + translator_cost,
        translator_cost=translator_cost,
        translated=proxied.headers.get("x-lingua-translated") == "true",
        latency_ms=(time.monotonic() - started) * 1000,
        detail={
            "baseline_input": baseline_usage.input_tokens,
            "baseline_output": baseline_usage.output_tokens,
            "proxied_input": proxied_usage.input_tokens,
            "proxied_output": proxied_usage.output_tokens,
            "truncated": native.json().get("stop_reason") == "max_tokens"
            or proxied.json().get("stop_reason") == "max_tokens",
        },
    )


def run_live_bench(
    *,
    upstream: str,
    mode: str = "estimate",
    model: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Run every corpus prompt through an in-process proxy against ``upstream``.

    The translator's fee is read from the cost row the proxy writes for each
    proxied request, so the comparison charges for every token on both paths.
    """
    import asyncio
    import tempfile

    model = model or os.environ.get("LINGUA_BENCH_MODEL", DEFAULT_MODEL)
    cost_log = pathlib.Path(tempfile.mkdtemp()) / "bench_cost.jsonl"
    settings = Settings(upstream_anthropic_url=upstream, memo_persist=False, cost_log_path=cost_log)
    app = create_app(settings, transport=transport)
    app.state.bench_cost_log = cost_log
    codec = AnthropicMessagesCodec()

    async def go() -> list[BenchResult]:
        results = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://bench.test"
        ) as client:
            client._lingua_cost_log = cost_log
            for row in load_corpus():
                results.append(await _one(client, row, model, codec))
        return results

    return build_report(asyncio.run(go()))
