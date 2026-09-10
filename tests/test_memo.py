"""The memo is what makes repeated history cheap and cache-stable.

Agentic clients resend the whole conversation every turn. Without a memo we
would re-translate all of it, non-deterministically, breaking the upstream
prompt cache and paying again each turn.
"""

from __future__ import annotations

import asyncio
import json

from lingua_proxy.memo import Memo


def test_stores_and_retrieves_forward_direction(tmp_path):
    memo = Memo(path=tmp_path / "memo.jsonl")
    memo.put("안녕하세요", "Hello", source="ko", target="en")

    hit = memo.get("안녕하세요")
    assert hit is not None
    assert hit.text == "Hello"
    assert hit.source == "ko"
    assert hit.target == "en"


def test_stores_the_reverse_direction_too(tmp_path):
    """The reply we hand back must be swappable to the model's own English."""
    memo = Memo(path=tmp_path / "memo.jsonl")
    memo.put("Hello there", "안녕하세요", source="en", target="ko")

    hit = memo.get("안녕하세요")
    assert hit is not None and hit.text == "Hello there"


def test_miss_returns_none(tmp_path):
    memo = Memo(path=tmp_path / "memo.jsonl")
    assert memo.get("본 적 없는 문장") is None


def test_persists_to_jsonl_and_reloads(tmp_path):
    path = tmp_path / "memo.jsonl"
    first = Memo(path=path)
    first.put("안녕하세요", "Hello", source="ko", target="en")

    reloaded = Memo(path=path)
    hit = reloaded.get("안녕하세요")
    assert hit is not None and hit.text == "Hello"


def test_jsonl_line_shape_is_stable(tmp_path):
    path = tmp_path / "memo.jsonl"
    memo = Memo(path=path)
    memo.put("안녕", "Hi", source="ko", target="en", model="claude-haiku-4-5")

    line = json.loads(path.read_text().strip())
    assert line["v"] == 1
    assert line["src"] == "안녕"
    assert line["dst"] == "Hi"
    assert line["sl"] == "ko"
    assert line["tl"] == "en"
    assert "ts" in line


def test_file_permissions_are_owner_only(tmp_path):
    """The memo holds prompt text, so it must not be world readable."""
    path = tmp_path / "memo.jsonl"
    memo = Memo(path=path)
    memo.put("안녕", "Hi", source="ko", target="en")

    assert (path.stat().st_mode & 0o077) == 0


def test_truncated_last_line_is_tolerated(tmp_path):
    """A crash mid-append must not make the whole memo unreadable."""
    path = tmp_path / "memo.jsonl"
    good = json.dumps({"v": 1, "src": "안녕", "dst": "Hi", "sl": "ko", "tl": "en", "ts": 1})
    path.write_text(good + "\n" + '{"v": 1, "src": "잘림", "ds')

    memo = Memo(path=path)
    assert memo.get("안녕").text == "Hi"
    assert memo.get("잘림") is None


def test_persist_disabled_keeps_everything_in_memory(tmp_path):
    path = tmp_path / "memo.jsonl"
    memo = Memo(path=path, persist=False)
    memo.put("안녕", "Hi", source="ko", target="en")

    assert memo.get("안녕").text == "Hi"
    assert not path.exists()


def test_whitespace_variant_hits_and_reattaches_original_whitespace(tmp_path):
    """Same sentence, different surrounding whitespace, must not re-translate."""
    memo = Memo(path=tmp_path / "memo.jsonl")
    memo.put("안녕하세요", "Hello", source="ko", target="en")

    hit = memo.get("\n  안녕하세요  \n")
    assert hit is not None
    assert hit.text == "\n  Hello  \n"


def test_exact_match_is_preferred_over_normalized(tmp_path):
    memo = Memo(path=tmp_path / "memo.jsonl")
    memo.put("  안녕  ", "PADDED", source="ko", target="en")
    memo.put("안녕", "BARE", source="ko", target="en")

    assert memo.get("  안녕  ").text == "PADDED"


def test_lru_evicts_the_least_recently_used(tmp_path):
    memo = Memo(path=tmp_path / "memo.jsonl", max_entries=4)
    memo.put("하나", "one", source="ko", target="en")
    memo.put("둘", "two", source="ko", target="en")

    memo.get("하나")  # refresh the first entry
    memo.put("셋", "three", source="ko", target="en")

    assert memo.get("하나") is not None, "recently used entry was evicted"


def test_eviction_does_not_corrupt_remaining_entries(tmp_path):
    memo = Memo(path=tmp_path / "memo.jsonl", max_entries=4)
    for i in range(10):
        memo.put(f"문장{i}", f"sentence{i}", source="ko", target="en")

    hit = memo.get("문장9")
    assert hit is not None and hit.text == "sentence9"


async def test_concurrent_identical_misses_translate_once(tmp_path):
    """Two turns arriving together must not both pay for the same translation."""
    memo = Memo(path=tmp_path / "memo.jsonl")
    calls: list[str] = []

    async def translate(text: str) -> str:
        calls.append(text)
        await asyncio.sleep(0.02)
        return "Hello"

    async def worker() -> str:
        async with memo.inflight("안녕하세요") as slot:
            if slot.done:
                return slot.value
            value = await translate("안녕하세요")
            slot.set(value)
            return value

    results = await asyncio.gather(worker(), worker(), worker())

    assert results == ["Hello", "Hello", "Hello"]
    assert len(calls) == 1, f"translated {len(calls)} times instead of once"


def test_stats_report_hits_and_misses(tmp_path):
    memo = Memo(path=tmp_path / "memo.jsonl")
    memo.put("안녕", "Hi", source="ko", target="en")
    memo.get("안녕")
    memo.get("없음")

    assert memo.hits == 1
    assert memo.misses == 1
