# lingua-proxy

[![CI](https://github.com/hanyoungYoo/lingua-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/hanyoungYoo/lingua-proxy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A local reverse proxy that sits between your LLM client and the API, translates
non-English prompts to English before they are sent, and translates the answer
back into your language.

Korean, Japanese, Chinese, Arabic and other non-Latin scripts tokenize two to
four times more expensively than English. lingua-proxy pays a small translation
fee on a cheap model to avoid that penalty on an expensive one.

> **Status: v0.1, alpha.** Read the [Honest cost model](#honest-cost-model) and
> [Caveats](#caveats) before pointing real traffic at it. It does not pay off
> for every workload, and this README says exactly when it does not.

## Quickstart

```bash
pip install lingua-proxy
```

```bash
lingua-proxy wrap claude
```

`wrap` starts the proxy, launches Claude Code pointed at it, and restores your
settings when you exit. Nothing is left behind.

Other commands:

```bash
lingua-proxy proxy --port 8787
```

```bash
lingua-proxy doctor
```

```bash
lingua-proxy dashboard
```

```bash
lingua-proxy bench
```

## How it works

```
client ──► lingua-proxy ──► your upstream (API or gateway)
             │  detect language (offline, no API call)
             │  English ────────────────► pass through untouched
             │  other ──► translate to English ──► forward
             └─────────── translate reply back ◄───
```

Language detection is offline and deterministic. English input is a pure
passthrough with no added latency and no extra tokens.

A **translation memo** stores every segment it has translated, in both
directions. Agentic clients resend the whole conversation on every turn, so the
memo keeps previously translated text byte-identical across turns. That
preserves the upstream prompt cache and means you pay the translation fee once
per unique piece of text, not once per turn.

## What is and isn't translated

Translated:

- Your prompts, when they are not already English.
- The assistant's text replies, back into your language.

Never touched:

- Code fences, inline code, URLs, file paths and JSON blobs, which are masked
  out before translation and restored afterwards byte-for-byte.
- System prompts, tool definitions, tool results, images and documents.
- Model reasoning/thinking blocks.
- Requests whose model matches `translate.skip_models` (by default the small
  helper models an agent calls in the background).
- Any request carrying the header `x-lingua-bypass: true`.

## Honest cost model

Savings come from the difference in how the same meaning tokenizes across
languages, minus what the translation itself costs.

**Where the saving actually is.** Translation shrinks your *input* tokens. It
does not shrink the reply, and output tokens cost roughly five times more per
token than input. So the headline percentage depends heavily on how much your
workload reads versus how much it writes.

Measured input-token savings, 18 prompts against a real Sonnet-class model:

| Language | Input tokens saved |
| --- | --- |
| Arabic | 59.0% |
| Korean | 51.7% |
| Chinese | 33.5% |
| Japanese | 32.2% |
| **All languages** | **44.6%** |

The raw per-prompt counts are committed in `tests/fixtures/bench_recorded.json`,
and a test asserts this README quotes the same figure, so the number cannot drift
away from the measurement.

Short prompts save nothing at all, by design: below the detection threshold the
proxy declines to translate rather than risk mangling a fragment.

| Workload | Verdict |
| --- | --- |
| Long non-English prompts, short replies | Pays off most |
| Ordinary non-English chat and Q&A | Pays off |
| Long-form generation (short prompt, long reply) | Marginal, the reply dominates |
| Code-heavy payloads | Can cost more than it saves |
| Very short prompts | No effect, translation is skipped |
| Quality-critical or legal text | Not recommended at any price |

Because the reply is the expensive half, an end-to-end dollar figure is lower
than the input-token figure above. Measure your own traffic rather than trusting
either number.

Run the benchmark yourself, against your own upstream and your own model:

```bash
lingua-proxy bench --mode estimate --multiturn
```

It reports dollars, not just tokens, because the translator model is priced
differently from your main model. It labels each category `pays off`,
`marginal`, or `loses`, and it is designed to be able to tell you that the
answer is no.

## Caveats

**Streaming.** In v0.1 each text block is buffered, translated, then emitted.
Tool calls and reasoning stream through untouched, and keep-alives are sent
while a translation is in flight, but you do not see non-English text appear
word by word. Sentence-boundary streaming translation is the planned fix before
1.0.

**Translation quality.** A cheap model does the translating. Technical nuance
can shift. Markdown tables are best-effort. If a translation fails or mangles a
protected placeholder, the request falls back to passing your original text
through untranslated rather than sending you something wrong.

That covers translations that *break*. It does not cover translations that are
fluent and simply wrong — a dropped negation turning "do not delete this" into
"delete this" passes every structural check there is. No automated check can
catch that, so the proxy does not pretend to. See
[When a translation is wrong](#when-a-translation-is-wrong).

**Prompt caching.** The proxy rewrites conversation history, so cache stability
depends on the memo. Deleting the memo file, or letting entries age out of a
long conversation, causes a one-time cache miss on the next turn.

**Token counts in your client.** Your client reports the tokens of the English
request it did not write. That is the point, but it can look surprising.

**Privacy.** The memo stores your prompts and the replies in plaintext on your
own machine, at `~/.lingua-proxy/memo.jsonl`, mode `600`. Set
`memo.persist = false` to keep it in memory only. The cost log stores counts,
never text. The proxy binds to loopback only.

**Credentials.** By default the translator reuses the same upstream and the
same credential your client already sends, so there is no second key to manage.
If your credential is a personal subscription token, check that automated
translation calls are within its terms, or set `translator.api_key` to a
separate key.

## When a translation is wrong

A translation can be grammatical, well-formed, and still mean the wrong thing.
This is the real risk of putting a model in the middle of your prompts, and it
is worth being direct: **the proxy cannot detect it.** Every check it performs
is structural — did the request succeed, did the segments come back, did the
code placeholders survive. A confidently inverted negation passes all of them.

So the defence is visibility rather than detection.

**Every translated response says so.** Three headers come back on each reply:

| Header | Meaning |
| --- | --- |
| `x-lingua-translated` | `true` or `false` |
| `x-lingua-source-lang` | the language that was detected |
| `x-lingua-prompt-en` | the English the model actually received, percent-encoded |

If an answer looks like it addressed a different question, that third header
tells you whether the model misunderstood you or the proxy mistranslated you.

**Check before you spend.** Send `x-lingua-review: true` and the proxy returns
the translation without calling the model at all:

```bash
curl -s http://127.0.0.1:8787/v1/messages   -H "x-lingua-review: true" -H "content-type: application/json"   -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"이 함수를 삭제하지 말고 설명만 해줘"}]}'
```

```json
{
  "lingua_review": true,
  "would_translate": true,
  "source_lang": "ko",
  "translated_prompt": "Don't delete this function, just explain it."
}
```

**Skip translation when it matters.** Any request carrying
`x-lingua-bypass: true` goes through untouched. For work where a subtle shift
in meaning is unacceptable — legal text, medical content, exact quotations,
anything you would not want paraphrased — use the bypass header, or do not
route it through this proxy at all.

**Keep a record.** Set `audit_log_path` in your config to log both sides of
every rewrite:

```toml
audit_log_path = "~/.lingua-proxy/audit.jsonl"
```

Each line holds the original, the translation, and the direction, so a
mistranslation can be found after the fact. It is off by default because,
unlike the cost log, it necessarily contains your prompt text.

**What this means in practice.** Translation is a lossy transformation applied
to your words. It is a good trade for ordinary conversational prompts, where a
slight rewording changes nothing. It is a poor trade when the exact wording is
the point. The proxy gives you the tools to see what it did and to turn it off
per request; deciding which of your traffic can tolerate it is yours to make.

## Using it behind an existing gateway

lingua-proxy chains. If your client already points at a company LLM gateway,
LiteLLM, a self-hosted relay, or another local proxy, lingua-proxy discovers
that base URL, keeps it as its own upstream, and inserts itself in front.

Path prefixes and non-default ports are preserved, and both credential styles
(`Authorization: Bearer` and `x-api-key`) are forwarded untouched. Set it
explicitly if you prefer:

```bash
lingua-proxy proxy --upstream https://gateway.example.com/anthropic/
```

There are no private endpoints baked into this package. The shipped defaults
are the public APIs.

## Endpoints

| Path | Behaviour |
| --- | --- |
| `POST /v1/messages` | Translated (Anthropic Messages format) |
| `POST /v1/chat/completions` | Translated (OpenAI-compatible format) |
| `GET /healthz` | Liveness and identity |
| `GET /stats` | Savings summary as JSON |
| anything else | Transparent passthrough |

## Requirements

Python 3.12 or newer. macOS and Linux. Windows support is on the roadmap.

## Why a proxy and not a plugin

A plugin lives inside one client and depends on that client's hook API, so it
needs separate work for every tool and cannot see the actual request body on
the wire. A proxy sits on the connection itself: it works with any client that
can point at a base URL, and it can transform the payload, which is exactly
what translation requires.

## Roadmap

- Sentence-boundary streaming translation
- DeepL and LibreTranslate adapters (the interface exists in v0.1)
- Optional system-prompt translation
- Windows support

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for how to set up, and for the three design rules that shape the code: fail open,
keep resent history byte-stable, and never hard-code a private endpoint.

Security issues should be reported privately. See [SECURITY.md](SECURITY.md),
which also describes exactly what data this proxy touches.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE)
