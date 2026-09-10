"""Language detection decides whether we translate at all.

The bias is deliberate and one-directional: when in doubt, do not translate.
A false negative costs a missed saving; a false positive corrupts a user's
prompt and spends money doing it.
"""

from __future__ import annotations

import pytest

from lingua_proxy.detector import Detector
from lingua_proxy.segments import mask

KOREAN = "이 함수가 왜 느린지 설명해 주고, 더 빠르게 만들 수 있는 방법을 알려줘."
JAPANESE = "この関数が遅い理由を説明して、速くする方法を教えてください。"
CHINESE = "请解释一下这个函数为什么运行得这么慢，并给出优化建议。"
ARABIC = "اشرح لي لماذا هذه الدالة بطيئة وكيف يمكن تسريعها."
ENGLISH = "Explain why this function is slow and how to make it faster."


@pytest.fixture(scope="module")
def detector() -> Detector:
    # Building the detector preloads language models, so share it.
    return Detector()


@pytest.mark.parametrize(
    "text,expected",
    [(KOREAN, "ko"), (JAPANESE, "ja"), (CHINESE, "zh"), (ARABIC, "ar")],
)
def test_non_english_is_detected_with_iso_code(detector, text, expected):
    decision = detector.detect(text)
    assert decision.translate is True
    assert decision.lang == expected


def test_english_is_passthrough(detector):
    decision = detector.detect(ENGLISH)
    assert decision.translate is False
    assert decision.lang == "en"


def test_short_text_is_not_translated(detector):
    """Detection is unreliable on fragments, so we leave them alone."""
    decision = detector.detect("안녕")
    assert decision.translate is False


def test_mixed_text_with_a_korean_clause_is_korean(detector):
    text = "Explain why parse_config is slow. 참고: 이 코드는 프로덕션에서 돌아가고 있어."
    decision = detector.detect(text)
    assert decision.translate is True
    assert decision.lang == "ko"


def test_english_sentence_mentioning_a_few_cjk_characters_stays_english(detector):
    """A stray CJK word must not flip a long English sentence."""
    text = "The Korean word 안녕 means hello, and this sentence is otherwise entirely English."
    decision = detector.detect(text)
    assert decision.translate is False
    assert decision.lang == "en"


def test_code_only_input_is_not_translated(detector):
    code = "def add(a, b):\n    return a + b\n"
    assert detector.detect(code).translate is False


def test_masked_code_block_with_korean_prose_detects_korean(detector):
    """The detector sees prose only, so identifiers cannot outvote the prose."""
    text = (
        "```python\ndef compute_statistics(values, threshold):\n    return sum(values)\n```\n"
        + KOREAN
    )
    decision = detector.detect(mask(text).prose)
    assert decision.translate is True
    assert decision.lang == "ko"


def test_empty_and_whitespace_are_not_translated(detector):
    assert detector.detect("").translate is False
    assert detector.detect("   \n  ").translate is False


def test_symbols_and_numbers_only_are_not_translated(detector):
    assert detector.detect("1234 5678 !!! ??? ---- ====").translate is False


def test_decision_reports_confidence(detector):
    decision = detector.detect(KOREAN)
    assert 0.0 < decision.confidence <= 1.0


def test_detection_is_deterministic(detector):
    assert [detector.detect(KOREAN).lang for _ in range(5)] == ["ko"] * 5


def test_language_set_is_configurable(detector):
    """A language outside the configured set must not be reported."""
    narrow = Detector(languages=("en", "ko"))
    assert narrow.detect(JAPANESE).lang != "ja"


def test_unknown_language_code_in_config_is_ignored():
    d = Detector(languages=("en", "ko", "not-a-language"))
    assert d.detect(KOREAN).lang == "ko"
