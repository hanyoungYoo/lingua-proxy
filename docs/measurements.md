# Measurements

How the numbers in the [README](../README.md) were produced, what they exclude,
and where the raw counts live.

Two runs are recorded, measuring different things:

| Run | Fixture | Measures |
| --- | --- | --- |
| Input-token bench, 2026-09-10 | `tests/fixtures/bench_recorded.json` | Input tokens only, 18 prompts, 4 languages |
| Head to head, 2026-09-15 | `tests/fixtures/head_to_head.json` | End-to-end dollars, every token charged |

The head-to-head run is the one that decides whether the proxy pays off. The
input-token bench measures one leg of the ledger and is reported separately
because it is not, on its own, a reason to adopt anything.

Tests in `tests/test_bench.py` assert that both this document and the README
quote these fixtures accurately, so the prose cannot drift away from the data.

## Head to head: does it pay off?

**Method.** Each prompt is sent twice. Path A sends the non-English prompt
straight to the model, which answers in the same language. Path B translates the
prompt to English, the model answers in English, and the answer is translated
back. Every token on both paths is charged, including the translator's own
calls, read from the API's usage fields rather than estimated.

Sonnet 4.6 as the main model, Haiku 4.5 as the translator, one three-sentence
question per language, temperature 0.

**Results.** Six cases, each language run with formatting preservation on and
off:

| Language | Formatting | Main model, native | Main model, English | Translator in/out | Saved |
| --- | --- | --- | --- | --- | --- |
| Korean | off | 351 | 106 | 408 / 281 | 36% cheaper |
| Korean | on | 387 | 190 | 492 / 449 | 0% |
| Japanese | off | 308 | 156 | 435 / 272 | 11% cheaper |
| Japanese | on | 280 | 158 | 451 / 278 | 6% more expensive |
| Chinese | off | 795 | 360 | 636 / 511 | 28% cheaper |
| Chinese | on | 892 | 329 | 610 / 459 | 39% cheaper |

Token counts are output tokens for the main model, and input/output for the
translator. Dollar figures per case are in the fixture under
`true_fee_measurement`.

**Caveat, and it is a large one.** Four of these six replies hit `max_tokens`.
A truncated reply is pinned to the same length in both languages, so it cannot
shrink and the fee buys nothing. Real-world savings depend on answers finishing
naturally, which means these six numbers are closer to a floor than to a typical
case — but they are also single samples, so treat the range as the shape of the
answer rather than its precision.

### Why the translator's fee is so large

Haiku writes Korean at about 0.9 characters per token, and Chinese and Japanese
at about 1.1 — the same rate the expensive model pays. It is cheaper only
because its price per token is roughly a third of Sonnet's, not because it
handles the language more efficiently. The fee therefore scales with the length
of the reply, exactly as the saving does.

Each translator call also carries about 350 tokens of fixed prompt overhead, so
very small requests lose regardless of how well they compress.

### Why an earlier version overstated the savings

Earlier runs reported 58–69% for Korean. Those runs estimated the translator's
fee by dividing character counts by three, a rule of thumb borrowed from
English. Korean encodes at close to one character per token, so the fee for
writing the reply back was undercounted roughly threefold, and the savings were
correspondingly inflated.

The 2026-09-15 run captures Haiku's real usage from the API for every translator
call. It supersedes the `cases` and `followup_bounded_untruncated` entries in the
fixture, which are kept only so the correction is auditable. The shipped `bench`
command now charges the translator's fee too; before this version it compared
the expensive model's usage alone.

## Input tokens: 18 prompts, 4 languages

**Method.** Each prompt is sent twice through the proxy, once with
`x-lingua-bypass` (native language) and once translated, and the input tokens
are compared. Output tokens are excluded here because reply length varies
independently and truncation at `max_tokens` pins both runs to the same count.

Recorded 2026-09-10 against a Sonnet-class model with a Haiku translator.

| Language | Prompts | Native input | Proxied input | Saved |
| --- | --- | --- | --- | --- |
| Arabic | 3 | 212 | 87 | 59.0% |
| Korean | 5 | 356 | 172 | 51.7% |
| Chinese | 5 | 230 | 153 | 33.5% |
| Japanese | 5 | 242 | 164 | 32.2% |
| **All** | **18** | **1040** | **576** | **44.6%** |

**What this does not show.** Input-token savings only. Output tokens cost
roughly five times more per token, and this run does not charge the translator
at all. A reader who stops here will overestimate the benefit; the head-to-head
run above is the honest ledger.

The 18 prompts split into 12 conversational, 3 short, and 3 long-form. The three
short prompts show exactly 0% by design: below the detection threshold the proxy
declines to translate rather than risk mangling a fragment.

## Is the saving just summarization?

If the English answer is shorter, perhaps the saving is only that you received a
thinner answer. Tested directly, the result splits into a real effect and a real
cost.

**Controlled: the compression is genuine.** The same question was asked in
Korean and in English with an identical explicit format instruction — a markdown
numbered list, exactly three items, one sentence each — so neither side could
win by saying less:

| | Output tokens | Characters delivered | Chars per token |
| --- | --- | --- | --- |
| Korean | 273 | 286 | 1.05 |
| English | 96 | 410 | 4.27 |

English delivered 43% more text using 65% fewer tokens. A summarization effect
would have produced less text, not more. The 4.1x gap is how the tokenizer
encodes each script, which is the premise of the project.

**Uncontrolled: some of it really is content loss.** In a free-form comparison
the round-tripped Korean answer delivered 37% fewer characters than the native
one. All five substantive facts survived, but it lost a concrete example and
every piece of markdown structure — the heading and the numbered list became a
wall of prose.

So translation genuinely compresses, *and* the round trip can flatten
formatting. The second part is fixable.

## Restoring the answer's shape

**Method.** The translator was never the problem; it preserves markdown
faithfully. The loss comes from the model formatting differently when it answers
in English. So the fix is an instruction to the model, not to the translator.
Same question, three paths:

| Path | Output tokens | Chars | Heading | List |
| --- | --- | --- | --- | --- |
| Native Korean | 313 | 354 | yes | yes |
| Proxied, instruction off | 104 | 237 | no | no |
| Proxied, instruction on | 200 | 458 | yes | bullets |

The instruction restores the heading and delivers 458 characters, more than the
native answer's 354, while still using 36% fewer output tokens. Bullets replaced
the numbered list, so the shape is close but not identical.

**Cost.** Savings drop from 67% to 36% of native output tokens. That is the
price of not silently flattening the answer, and it is why the instruction is on
by default: quietly degrading output is not a trade anyone opted into.

### Tuning the instruction

A second version of the instruction names the mirroring rule explicitly — if the
question asks for N points, use a numbered list. Shapes observed, where
`H`=heading, `N`=numbered list, `B`=bullets:

| Prompt | Native | First version | Current |
| --- | --- | --- | --- |
| ko-list | `HN-` | `H-B` | `-N-` |
| ko-steps | `H-B` | `H-B` | `H-B` |
| ja-list | `-N-` | `H-B` | `-N-` |

The current version matched the native shape on 2 of 3 prompts against 1 of 3
for the first, and used fewer tokens doing it.

**Why exact matching is the wrong target.** The native answer is itself
inconsistent — it uses a heading for the Korean list question and none for the
same question in Japanese. The goal is a structured answer, not a byte-identical
one.

## Reproducing this

The fixtures contain counts only: no prompt text, no endpoints, no credentials.
Tests assert that, so they stay shareable.

None of these runs are a substitute for measuring your own traffic:

```bash
lingua-proxy bench --mode estimate --multiturn
```

It reports dollars rather than tokens, because the translator is priced
differently from the main model, and labels each category `pays off`,
`marginal`, or `loses`.
