"""Savings accounting in dollars.

Tokens are the mechanism; money is the point. The translator runs on a cheap
model and the main request on an expensive one, so the arithmetic has to price
both or it will happily report a "saving" that costs more than it saves.

Every row therefore carries what actually happened (measured usage) alongside
what would have happened untranslated (the counterfactual), and a verdict that
is allowed to say ``loses``.

The log holds counts only -- never prompt or reply text -- so it can be shared
or pasted into an issue. Text lives in the memo, which is private.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

from lingua_proxy.codecs import Usage

#: Cache reads bill at a fraction of the input rate; cache writes at a premium.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices for one model."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: Usage) -> float:
        million = 1_000_000
        return (
            usage.input_tokens * self.input_per_mtok
            + usage.cache_read_input_tokens * self.input_per_mtok * CACHE_READ_MULTIPLIER
            + usage.cache_creation_input_tokens * self.input_per_mtok * CACHE_WRITE_MULTIPLIER
            + usage.output_tokens * self.output_per_mtok
        ) / million


#: Published list prices. Override in config for gateways that bill differently.
PRICES: dict[str, ModelPrice] = {
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "claude-sonnet-5": ModelPrice(2.0, 10.0),
    "claude-opus-4-8": ModelPrice(5.0, 25.0),
    "claude-opus-5": ModelPrice(5.0, 25.0),
    "gpt-4o-mini": ModelPrice(0.15, 0.6),
    "gpt-4o": ModelPrice(2.5, 10.0),
}

_FALLBACK_PRICE = ModelPrice(3.0, 15.0)


def price_for(model: str) -> ModelPrice:
    """Look up a price, tolerating dated snapshot suffixes."""
    if not model:
        return _FALLBACK_PRICE
    if model in PRICES:
        return PRICES[model]
    for name, price in PRICES.items():
        if model.startswith(name):
            return price
    return _FALLBACK_PRICE


#: Rough characters-per-token by script, used only when an exact count is
#: unavailable. Non-Latin scripts pack fewer characters per token, which is
#: the entire reason this proxy exists.
_CHARS_PER_TOKEN_LATIN = 4.0
_CHARS_PER_TOKEN_OTHER = 1.1


def estimate_tokens(text: str) -> int:
    """Estimate tokens for text when a real count is not available."""
    if not text:
        return 0

    latin = other = 0
    for ch in text:
        if ch.isspace():
            continue
        name = unicodedata.name(ch, "")
        if any(script in name for script in ("HANGUL", "CJK", "HIRAGANA", "KATAKANA", "ARABIC")):
            other += 1
        else:
            latin += 1

    return max(1, round(latin / _CHARS_PER_TOKEN_LATIN + other / _CHARS_PER_TOKEN_OTHER))


@dataclass
class CostRow:
    """One request's accounting."""

    model: str = ""
    translator_model: str = ""
    source_lang: str | None = None
    endpoint: str = ""
    translated: bool = False
    usage: Usage = field(default_factory=Usage)
    translator_usage: Usage = field(default_factory=Usage)
    counterfactual_input: int = 0
    counterfactual_output: int = 0
    estimate_method: str = "heuristic"
    translator_calls: int = 0
    memo_hits: int = 0
    memo_miss_assistant: int = 0
    fallback_reason: str | None = None
    skipped_reason: str | None = None
    latency_ms: float = 0.0

    # Held in memory for the counterfactual only; never written to disk.
    english_texts: list[str] = field(default_factory=list, repr=False)
    translated_texts: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # A row carrying counterfactual data was translated by definition.
        # Without this, forgetting the flag silently reports zero savings.
        if self.counterfactual_input or self.counterfactual_output:
            self.translated = True

    @property
    def proxied_cost(self) -> float:
        """What this request actually cost, translator included."""
        main = price_for(self.model).cost(self.usage)
        translator = price_for(self.translator_model).cost(self.translator_usage)
        return main + translator

    @property
    def baseline_cost(self) -> float:
        """What the same request would have cost untranslated."""
        if not self.translated:
            return self.proxied_cost

        price = price_for(self.model)
        measured_input = self.usage.total_input or 1
        # Keep the measured cache mix so the baseline is not unfairly priced
        # as entirely uncached.
        scale = self.counterfactual_input / measured_input
        baseline_usage = Usage(
            input_tokens=round(self.usage.input_tokens * scale),
            cache_read_input_tokens=round(self.usage.cache_read_input_tokens * scale),
            cache_creation_input_tokens=round(self.usage.cache_creation_input_tokens * scale),
            output_tokens=self.counterfactual_output,
        )
        return price.cost(baseline_usage)

    @property
    def dollars_saved(self) -> float:
        return self.baseline_cost - self.proxied_cost

    @property
    def savings_ratio(self) -> float:
        if not self.translated or self.baseline_cost <= 0:
            return 0.0
        return self.dollars_saved / self.baseline_cost

    @property
    def verdict(self) -> str:
        if not self.translated:
            return "passthrough"
        ratio = self.savings_ratio
        if ratio >= 0.25:
            return "pays off"
        if ratio > 0:
            return "marginal"
        return "loses"

    def to_json(self) -> dict:
        """Serialize counts only. No prompt or reply text, ever."""
        return {
            "v": 1,
            "ts": int(time.time()),
            "model": self.model,
            "translator_model": self.translator_model,
            "source_lang": self.source_lang,
            "endpoint": self.endpoint,
            "translated": self.translated,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_read_input_tokens": self.usage.cache_read_input_tokens,
            "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
            "translator_input_tokens": self.translator_usage.input_tokens,
            "translator_output_tokens": self.translator_usage.output_tokens,
            "counterfactual_input": self.counterfactual_input,
            "counterfactual_output": self.counterfactual_output,
            "estimate_method": self.estimate_method,
            "translator_calls": self.translator_calls,
            "memo_hits": self.memo_hits,
            "memo_miss_assistant": self.memo_miss_assistant,
            "fallback_reason": self.fallback_reason,
            "skipped_reason": self.skipped_reason,
            "latency_ms": round(self.latency_ms, 1),
            "proxied_cost": round(self.proxied_cost, 8),
            "baseline_cost": round(self.baseline_cost, 8),
        }

    @classmethod
    def from_json(cls, data: dict) -> CostRow:
        return cls(
            model=data.get("model", ""),
            translator_model=data.get("translator_model", ""),
            source_lang=data.get("source_lang"),
            endpoint=data.get("endpoint", ""),
            translated=bool(data.get("translated")),
            usage=Usage(
                input_tokens=int(data.get("input_tokens") or 0),
                output_tokens=int(data.get("output_tokens") or 0),
                cache_read_input_tokens=int(data.get("cache_read_input_tokens") or 0),
                cache_creation_input_tokens=int(data.get("cache_creation_input_tokens") or 0),
            ),
            translator_usage=Usage(
                input_tokens=int(data.get("translator_input_tokens") or 0),
                output_tokens=int(data.get("translator_output_tokens") or 0),
            ),
            counterfactual_input=int(data.get("counterfactual_input") or 0),
            counterfactual_output=int(data.get("counterfactual_output") or 0),
            estimate_method=data.get("estimate_method", "heuristic"),
            translator_calls=int(data.get("translator_calls") or 0),
            memo_hits=int(data.get("memo_hits") or 0),
            latency_ms=float(data.get("latency_ms") or 0.0),
        )


@dataclass
class Summary:
    """Aggregated savings across many requests."""

    requests: int = 0
    translated: int = 0
    passthrough: int = 0
    dollars_proxied: float = 0.0
    dollars_baseline: float = 0.0
    translator_dollars: float = 0.0
    by_language: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)

    @property
    def dollars_saved(self) -> float:
        return self.dollars_baseline - self.dollars_proxied

    @property
    def savings_ratio(self) -> float:
        if self.dollars_baseline <= 0:
            return 0.0
        return self.dollars_saved / self.dollars_baseline

    def to_json(self) -> dict:
        return {
            "requests": self.requests,
            "translated": self.translated,
            "passthrough": self.passthrough,
            "dollars_proxied": round(self.dollars_proxied, 6),
            "dollars_baseline": round(self.dollars_baseline, 6),
            "dollars_saved": round(self.dollars_saved, 6),
            "translator_dollars": round(self.translator_dollars, 6),
            "savings_ratio": round(self.savings_ratio, 4),
            "by_language": self.by_language,
            "by_model": self.by_model,
        }


def summarize(rows: Iterable[CostRow]) -> Summary:
    summary = Summary()
    for row in rows:
        summary.requests += 1
        if row.translated:
            summary.translated += 1
        else:
            summary.passthrough += 1
        summary.dollars_proxied += row.proxied_cost
        summary.dollars_baseline += row.baseline_cost
        summary.translator_dollars += price_for(row.translator_model).cost(row.translator_usage)

        if row.source_lang:
            summary.by_language[row.source_lang] = summary.by_language.get(row.source_lang, 0) + 1
        if row.model:
            summary.by_model[row.model] = summary.by_model.get(row.model, 0) + 1
    return summary


class CostLog:
    """Append-only JSONL log of per-request savings."""

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)

    def record(self, row: CostRow) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.to_json(), separators=(",", ":")) + "\n")
        if not existed:
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)

    def rows(self) -> list[CostRow]:
        """Read every intact row, skipping any damaged line."""
        try:
            raw = self.path.read_text()
        except OSError:
            return []

        out = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(CostRow.from_json(json.loads(line)))
            except ValueError:
                continue
        return out
