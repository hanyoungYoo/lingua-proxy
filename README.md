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

> **Status: v0.1, alpha, and it does not pay off for every workload.** With
> every token charged, including the translator's own, focused questions saved
> **0–39%** depending on language and settings, and sprawling or truncated
> replies lost money. An earlier version of this README overstated the savings;
> the [Honest cost model](#honest-cost-model) has the correction. Run
> `lingua-proxy bench` against your own traffic before adopting it.

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

**Where the saving actually is.** Both sides, but by different mechanisms.

*Input* is straightforward: your Korean prompt becomes a shorter English one.

*Output* is where most of the money is, and the effect there is larger but
easier to lose. Asked the same question, a model answering in English emits far
fewer tokens than one answering in Korean — measured at 115 versus 405 output
tokens for the same three-sentence explanation, a 72% reduction for content of
the same length. The proxy still hands you Korean, but the Korean is generated
by the cheap translation model rather than the expensive one:

| | Without proxy | With proxy |
| --- | --- | --- |
| Expensive model writes | 430 Korean tokens | 102 English tokens |
| Cheap model writes | — | the Korean you read |

That is the whole trick — and it also sets the condition under which it fails.

Both the saving and the translation fee scale with the length of the answer, so
**the length cancels out**. What decides profitability is only *how much shorter
the English answer is*. On a Sonnet-class model with a cheap translator, the
English reply must come out at least **38% shorter** to break even:

| Main model | English answer must be shorter by |
| --- | --- |
| Opus-class | 23% |
| Sonnet 4.6 | 38% |
| Sonnet 5 | 55% |
| Haiku | never pays — you would pay Haiku to rewrite Haiku |

Whether your traffic clears that bar is an empirical question, and the answer is
often no. See the head-to-head measurements below before assuming it does.
A reply that hits `max_tokens` is the worst case: pinned to the same length in
both languages, it cannot shrink at all, so the fee buys nothing.

Measured input-token savings, 18 prompts against a real Sonnet-class model.
These are real but they are only half the ledger: the input side always wins,
and the output side decides whether the request as a whole does.

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

### Measured head to head

Each prompt is sent twice: straight to the model in its own language, and
through the proxy. **Every token on both paths is charged, including the
translator's own calls, read from the API's usage fields.** Sonnet 4.6 with a
Haiku translator, one three-sentence question per language:

| Language | Formatting preserved (default) | Formatting off |
| --- | --- | --- |
| Korean | 0% | 36% cheaper |
| Japanese | 6% more expensive | 11% cheaper |
| Chinese | 39% cheaper | 28% cheaper |

Raw tokens for every call are in `tests/fixtures/head_to_head.json` under
`true_fee_measurement`.

**A correction.** Earlier versions of this section reported savings of 22–69%.
Those runs estimated the translator's fee from character counts divided by
three. Korean encodes at about one character per token, so the fee for writing
the reply back was undercounted roughly threefold and the savings were
overstated. The table above replaces them, and the shipped `bench` command now
charges the fee too — before this version it compared the expensive model's
usage alone.

**Why the fee is so large.** Haiku writes Korean at the same ~0.9 characters
per token that Sonnet does. It is cheaper only because its price per token is a
third of Sonnet's, not because it handles the language better. Each translator
call also carries about 350 tokens of fixed prompt, so very small requests lose
regardless of how well they compress.

**Why formatting preservation can erase the saving.** With it on, the
translated Korean reply came back *longer* than the native one — 449 tokens
against 387 — and Haiku writing those extra tokens cost exactly what Sonnet had
saved. Chinese was unaffected on this sample. This is a genuine quality-versus-
cost trade, not a free improvement; see
[Keeping the answer's shape](#keeping-the-answers-shape).

| Workload | Verdict |
| --- | --- |
| Focused questions, formatting off | Pays off modestly, 11–36% |
| Focused questions, formatting on | Language-dependent, 0–39% |
| Replies that hit `max_tokens` | Always loses; nothing can shrink |
| Open-ended "tell me everything" | Usually loses |
| Code-heavy payloads | Can cost more than it saves |
| Very short prompts | No effect, translation is skipped |
| Quality-critical or legal text | Not recommended at any price |

These are single samples at temperature 0. Treat them as the shape of the
answer, not its precision, and run `lingua-proxy bench` on your own traffic.

### Is this just summarization in disguise?

A fair challenge: if the English answer is shorter, maybe the saving is only
that you received a thinner answer. Tested directly, and the result splits.

**The compression is real.** Asked the same question in each language with an
identical format instruction (numbered list, exactly three items, one sentence
each), so neither side could win by saying less:

| | Output tokens | Characters delivered | Chars per token |
| --- | --- | --- | --- |
| Korean | 273 | 286 | 1.05 |
| English | 96 | 410 | 4.27 |

English delivered **43% more text using 65% fewer tokens**. Summarization would
have produced less text, not more. The 4.1x gap is how the tokenizer encodes
each script, which is the premise of this whole project.

**But some of it really is content loss.** In an uncontrolled comparison, the
round-tripped answer delivered 37% fewer characters than the native one. All
five substantive facts survived, but it lost a concrete example and every piece
of markdown structure — the heading and the numbered list became a wall of
prose.

So: translation genuinely compresses, *and* the round trip can flatten
formatting. The second part is fixable, and it is fixed by default.

### Keeping the answer's shape

The translator was never the problem — it preserves markdown faithfully. The
loss happens because a model answering in English formats differently than one
answering in Korean. So the fix is an instruction to the model, not to the
translator, asking it to mirror the structure the question implies.

It works: the heading returns and the reply carries more content than the
native answer. But charged honestly, it is not free:

| Korean, three-sentence question | Sonnet writes | Haiku writes | Net vs native |
| --- | --- | --- | --- |
| No proxy | 387 Korean tokens | — | — |
| Proxy, instruction off | 106 English | 281 Korean | 36% cheaper |
| Proxy, instruction on | 190 English | 449 Korean | 0% |

The structured reply is longer than the native one, and Haiku writing the
extra Korean consumed the whole saving on this sample. Chinese kept a 39%
saving with it on, so the cost is language- and prompt-dependent.

It is **on by default** because silently returning a thinner, flatter answer
is not a trade anyone opted into. But you should know you may be paying the
entire saving for it. Turn it off if you would rather have the tokens:

```toml
preserve_formatting = false
```

Or per request, with the header `x-lingua-preserve-formatting: false`.

### Which translator should I use?

Ask, rather than sweeping models at real cost:

```bash
lingua-proxy advise --model claude-sonnet-4-6
```

It computes the break-even threshold for every translator that shares your
upstream and recommends one. The answer is arithmetic, not an experiment: the
cheapest capable model always wins, because the fee is what eats the saving.
The bundled default is already that model.

The translator is the only setting that moves the economics. Detection is
offline and free, and translation runs at temperature 0 with no reasoning
enabled, so there is no effort or sampling knob to tune.

### Getting the most out of it

Since the saving comes from the expensive model writing English rather than
Korean, anything that shortens its reply increases the benefit:

- **Ask for bounded answers.** "In three sentences" or "as a short list" saves
  far more than the same question asked open-endedly.
- **Set `max_tokens` to a real limit, not a huge one.** A ceiling the answer
  actually hits guarantees zero shrink and a wasted fee.
- **Send bulk or code-heavy work through `x-lingua-bypass: true`.** Those are
  the cases that lose money.

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

**The answer can arrive flatter.** A model answering in English formats
differently than one answering in Korean, so a round-tripped reply used to lose
headings and lists. The proxy now asks the model to keep its usual formatting,
which restores the structure at the cost of roughly half the token saving. On by
default; disable with `preserve_formatting = false` or the header
`x-lingua-preserve-formatting: false`. See
[Keeping the answer's shape](#keeping-the-answers-shape)

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
