"""Bidirectional translation memo.

Agentic clients resend the entire conversation on every turn. Translating that
history afresh each time would be expensive, and worse, non-deterministic: the
bytes sent upstream would differ turn to turn, destroying the prompt cache and
costing *more* than not translating at all.

The memo fixes both. Every translated pair is stored in both directions:

* forward, so the same user sentence always produces the same English;
* reverse, so an assistant reply we translated can be swapped back to the
  model's original English when it reappears in history.

Storage is an append-only JSONL file in the user's home directory, mode 600,
because it contains prompt text.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    """A memo lookup that succeeded."""

    text: str
    source: str
    target: str


@dataclass
class Slot:
    """Coordination handle for a single in-flight translation."""

    done: bool = False
    value: str | None = None
    _future: asyncio.Future | None = None

    def set(self, value: str) -> None:
        self.value = value
        if self._future is not None and not self._future.done():
            self._future.set_result(value)


def _split_whitespace(text: str) -> tuple[str, str, str]:
    """Split into (leading whitespace, core, trailing whitespace)."""
    stripped = text.strip()
    if not stripped:
        return "", text, ""
    start = text.index(stripped[0])
    lead = text[:start]
    trail = text[start + len(stripped) :]
    return lead, stripped, trail


class Memo:
    """LRU translation cache with JSONL persistence."""

    def __init__(
        self,
        path: pathlib.Path,
        *,
        persist: bool = True,
        max_entries: int = 50_000,
    ):
        self.path = pathlib.Path(path)
        self.persist = persist
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

        self._entries: OrderedDict[str, Hit] = OrderedDict()
        self._inflight: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

        if self.persist and self.path.exists():
            self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        try:
            raw = self.path.read_text()
        except OSError:
            return

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # A crash mid-append leaves a partial last line. Skip it
                # rather than discarding an otherwise good memo.
                continue
            src, dst = record.get("src"), record.get("dst")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            sl = record.get("sl", "")
            tl = record.get("tl", "")
            self._insert(src, Hit(text=dst, source=sl, target=tl))
            self._insert(dst, Hit(text=src, source=tl, target=sl))

    def _append(self, record: dict) -> None:
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        if not existed:
            # Prompt text lives here; keep it owner-only.
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)

    # -- core ------------------------------------------------------------

    def _insert(self, key: str, hit: Hit) -> None:
        self._entries[key] = hit
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def put(
        self,
        source_text: str,
        translated_text: str,
        *,
        source: str,
        target: str,
        model: str | None = None,
        direction: str = "req",
    ) -> None:
        """Record a translated pair in both directions."""
        self._insert(source_text, Hit(text=translated_text, source=source, target=target))
        self._insert(translated_text, Hit(text=source_text, source=target, target=source))

        record = {
            "v": 1,
            "src": source_text,
            "dst": translated_text,
            "sl": source,
            "tl": target,
            "dir": direction,
            "ts": int(time.time()),
        }
        if model:
            record["model"] = model
        self._append(record)

    def get(self, text: str) -> Hit | None:
        """Look up a translation.

        Falls back to a whitespace-normalized match, re-attaching the caller's
        original surrounding whitespace, so the same sentence indented
        differently in a later turn still hits.
        """
        entry = self._entries.get(text)
        if entry is not None:
            self._entries.move_to_end(text)
            self.hits += 1
            return entry

        lead, core, trail = _split_whitespace(text)
        if core != text:
            entry = self._entries.get(core)
            if entry is not None:
                self._entries.move_to_end(core)
                self.hits += 1
                return Hit(
                    text=f"{lead}{entry.text}{trail}",
                    source=entry.source,
                    target=entry.target,
                )

        self.misses += 1
        return None

    # -- concurrency -----------------------------------------------------

    @contextlib.asynccontextmanager
    async def inflight(self, key: str) -> AsyncIterator[Slot]:
        """Ensure only one caller translates a given text at a time.

        Concurrent requests carrying the same sentence (a main call and a
        background helper call, say) would otherwise each pay for it.
        """
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is None:
                future: asyncio.Future = asyncio.get_running_loop().create_future()
                self._inflight[key] = future
                owner = True
            else:
                future, owner = existing, False

        if not owner:
            value = await future
            yield Slot(done=True, value=value)
            return

        slot = Slot(done=False, _future=future)
        try:
            yield slot
            if not future.done():
                future.set_result(slot.value)
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._inflight.pop(key, None)
