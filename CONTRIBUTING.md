# Contributing

Thanks for considering a contribution.

## Getting set up

Requires Python 3.12 or newer, and [uv](https://github.com/astral-sh/uv).

```bash
uv venv && uv pip install -e ".[dev]"
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

The whole suite runs offline in a few seconds. No test may reach the network:
upstreams are mocked with `RecordingTransport` in `tests/conftest.py`, and
translation is stubbed by `FakeTranslator` so results stay deterministic.

## How this project is built

Tests come first. Every module here was written by adding a failing test that
describes the behaviour, then the smallest implementation that satisfies it.
Please keep that shape — a pull request that changes behaviour should change or
add a test that would fail without it.

Three rules carry most of the design:

1. **Fail open.** Any uncertainty — a translator error, a damaged placeholder,
   an ambiguous language — must result in the user's original text being
   forwarded untranslated. A missed saving is a rounding error. A corrupted
   prompt is not.
2. **Byte stability.** Text already translated in an earlier turn is reproduced
   from the memo, never re-translated. Agentic clients resend the whole history
   each turn, so anything else breaks the upstream prompt cache and costs more
   than it saves.
3. **No private endpoints.** The only hard-coded URLs are the public vendor
   APIs. Everything else is resolved at runtime. `test_no_private_endpoints_committed_in_repo`
   fails the build if a private host is committed.

## Measuring cost claims

This project exists to make a claim about money, so please do not change the
documented savings figures by hand. Re-run the benchmark against a real upstream
and update the recorded fixture:

```bash
LINGUA_BENCH_BASE_URL=https://your-upstream.example lingua-proxy bench
```

A test asserts the README quotes the same number as
`tests/fixtures/bench_recorded.json`, so the two cannot drift apart. Note that
the benchmark spends real money and refuses to guess an endpoint for you.

## Reporting bugs

Please include the output of `lingua-proxy doctor`, which redacts credentials.
Never paste a prompt containing anything you would not publish — and note that
`~/.lingua-proxy/memo.jsonl` contains your prompts in plaintext, so do not
attach it.

## Areas that need work

See the roadmap in the README. The largest open item is sentence-boundary
streaming translation, which would remove the per-block buffering caveat.
