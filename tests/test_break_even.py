"""Whether translation pays depends on shrink ratio, not absolute length.

Both the saving and the translation fee scale linearly with the answer, so
length cancels out. What matters is how much shorter the English answer is
than the same answer in the user's language.
"""

from __future__ import annotations

import pytest

from lingua_proxy.cost_log import break_even_shrink, is_profitable, price_for


def test_break_even_is_a_ratio_not_a_length():
    """Doubling the answer changes nothing about whether it pays."""
    main, translator = price_for("claude-sonnet-4-6"), price_for("claude-haiku-4-5")

    for length in (100, 1_000, 100_000):
        assert is_profitable(0.30, main, translator, native_output=length) is True
        assert is_profitable(0.90, main, translator, native_output=length) is False


def test_break_even_for_sonnet_and_haiku_is_around_forty_percent():
    main, translator = price_for("claude-sonnet-4-6"), price_for("claude-haiku-4-5")
    ratio = break_even_shrink(main, translator)

    assert 0.30 < ratio < 0.45


def test_a_measured_short_answer_is_profitable():
    """Measured: 521 Korean tokens became 178 English, a 66% shrink."""
    main, translator = price_for("claude-sonnet-4-6"), price_for("claude-haiku-4-5")
    assert is_profitable(178 / 521, main, translator, native_output=521) is True


def test_a_measured_long_answer_is_not():
    """Measured: 836 Korean tokens became 693 English, only a 17% shrink."""
    main, translator = price_for("claude-sonnet-4-6"), price_for("claude-haiku-4-5")
    assert is_profitable(693 / 836, main, translator, native_output=836) is False


def test_a_truncated_answer_cannot_profit():
    """Both runs pinned to the ceiling means zero shrink and a fee for nothing."""
    main, translator = price_for("claude-sonnet-4-6"), price_for("claude-haiku-4-5")
    assert is_profitable(1.0, main, translator, native_output=8000) is False


def test_a_cheaper_translator_widens_the_profitable_range():
    main = price_for("claude-sonnet-4-6")
    cheap = price_for("claude-haiku-4-5")
    expensive = price_for("claude-sonnet-4-6")

    assert break_even_shrink(main, cheap) < break_even_shrink(main, expensive)


def test_translating_with_the_same_model_can_never_pay():
    """Paying the expensive model to rewrite its own answer is always a loss."""
    main = price_for("claude-sonnet-4-6")
    assert break_even_shrink(main, main) >= 1.0


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-opus-5"])
def test_pricier_main_models_are_easier_to_profit_from(model):
    """The bigger the gap to the translator, the easier translation pays."""
    translator = price_for("claude-haiku-4-5")
    sonnet = break_even_shrink(price_for("claude-sonnet-4-6"), translator)

    assert break_even_shrink(price_for(model), translator) < sonnet


def test_a_cheaper_main_model_is_harder_to_profit_from():
    """Sonnet 5 costs less per output token, so the margin is thinner."""
    translator = price_for("claude-haiku-4-5")

    assert break_even_shrink(price_for("claude-sonnet-5"), translator) > break_even_shrink(
        price_for("claude-sonnet-4-6"), translator
    )
