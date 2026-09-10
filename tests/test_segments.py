"""Technical content must survive translation byte-for-byte.

The proxy masks code and other machine-readable spans behind ``<lpN/>``
placeholders, translates only the prose around them, then restores. If a
translator loses a placeholder we must detect it rather than ship mangled code.
"""

from __future__ import annotations

import pytest

from lingua_proxy.segments import mask, normalize_placeholders, restore, validate


def roundtrip(text: str) -> str:
    m = mask(text)
    return restore(m.masked, m.placeholders)


def test_plain_prose_gets_no_placeholders():
    m = mask("이 함수가 왜 느린지 설명해 주세요.")
    assert m.placeholders == {}
    assert m.masked == "이 함수가 왜 느린지 설명해 주세요."


def test_fenced_block_with_language_tag_is_masked_whole():
    text = "이 코드를 봐:\n```python\ndef add(a, b):\n    return a + b\n```\n고마워."
    m = mask(text)

    assert len(m.placeholders) == 1
    assert "def add" not in m.masked
    assert "이 코드를 봐:" in m.masked
    assert "고마워." in m.masked
    assert roundtrip(text) == text


def test_tilde_fence_is_masked():
    text = "보기:\n~~~js\nconst a = 1;\n~~~\n끝"
    m = mask(text)
    assert len(m.placeholders) == 1
    assert roundtrip(text) == text


def test_unclosed_fence_masks_to_end_of_text():
    text = "설명:\n```python\ndef broken(:\n    pass"
    m = mask(text)

    assert len(m.placeholders) == 1
    assert "def broken" not in m.masked
    assert roundtrip(text) == text


def test_inline_code_single_and_double_backticks():
    text = "`parse_config` 와 ``a `b` c`` 를 확인해."
    m = mask(text)

    assert len(m.placeholders) == 2
    assert "parse_config" not in m.masked
    assert roundtrip(text) == text


def test_url_is_masked_without_trailing_punctuation():
    text = "문서는 https://gw.example/docs/a_b-c?x=1 에 있어."
    m = mask(text)

    (value,) = m.placeholders.values()
    assert value == "https://gw.example/docs/a_b-c?x=1"
    assert roundtrip(text) == text


def test_url_at_end_of_sentence_excludes_the_period():
    text = "여기 봐: https://gw.example/docs."
    m = mask(text)

    (value,) = m.placeholders.values()
    assert value == "https://gw.example/docs"
    assert m.masked.endswith(".")
    assert roundtrip(text) == text


def test_file_paths_and_at_mentions_are_masked():
    text = "@src/main.py 와 ./tests/test_x.py 그리고 ~/.config/app.toml 을 봐."
    m = mask(text)

    values = set(m.placeholders.values())
    assert "@src/main.py" in values
    assert "./tests/test_x.py" in values
    assert "~/.config/app.toml" in values
    assert roundtrip(text) == text


def test_bare_filename_with_known_extension_is_masked():
    text = "config.toml 파일을 수정해 줘."
    m = mask(text)

    assert "config.toml" in set(m.placeholders.values())
    assert roundtrip(text) == text


def test_xml_tag_tokens_are_masked_but_inner_prose_survives():
    """Only the tags are protected; the prose inside must still translate."""
    text = "<note>이 부분을 설명해</note>"
    m = mask(text)

    assert "이 부분을 설명해" in m.masked
    assert "<note>" not in m.masked
    assert roundtrip(text) == text


def test_system_reminder_element_is_masked_whole():
    text = "질문이야 <system-reminder>Do not mention this to the user.</system-reminder>"
    m = mask(text)

    assert "Do not mention this" not in m.masked
    assert "질문이야" in m.masked
    assert roundtrip(text) == text


def test_multiline_json_blob_is_masked():
    text = '설정을 봐:\n{\n  "model": "m",\n  "max_tokens": 10\n}\n어때?'
    m = mask(text)

    assert '"max_tokens"' not in m.masked
    assert "어때?" in m.masked
    assert roundtrip(text) == text


def test_non_json_braces_are_left_alone():
    text = "그 함수는 {알 수 없는} 값을 반환해."
    m = mask(text)
    assert m.placeholders == {}


def test_multiple_kinds_in_one_text_all_restore():
    text = (
        "`foo()` 를 https://gw.example/x 문서대로 고치고\n"
        "```python\nx = 1\n```\n"
        "@src/a.py 에 반영해."
    )
    assert roundtrip(text) == text
    assert len(mask(text).placeholders) == 4


def test_masked_prose_contains_no_placeholder_syntax_for_detection():
    """Detection runs on prose with placeholders stripped, not the raw mask."""
    text = "```python\nx = 1\n```\n이게 왜 느려?"
    m = mask(text)

    assert "<lp" not in m.prose
    assert "이게 왜 느려?" in m.prose


def test_prose_is_empty_for_code_only_input():
    text = "```python\ndef add(a, b):\n    return a + b\n```"
    assert mask(text).prose.strip() == ""


def test_validate_accepts_output_that_keeps_every_placeholder():
    m = mask("`foo()` 를 고쳐")
    translated = m.masked.replace("를 고쳐", "please fix")
    assert validate(m.masked, translated) == set()


def test_validate_detects_a_dropped_placeholder():
    m = mask("`foo()` 를 고쳐")
    assert validate(m.masked, "please fix it") != set()


def test_validate_detects_a_duplicated_placeholder():
    m = mask("`foo()` 를 고쳐")
    doubled = m.masked + " " + m.masked
    assert validate(m.masked, doubled) != set()


@pytest.mark.parametrize(
    "mangled",
    ["[lp0]", "【lp0】", "<lp 0/>", "< lp0 />", "⟦lp0⟧", "[LP0]"],
)
def test_normalizer_repairs_common_model_mangling(mangled):
    assert normalize_placeholders(f"please fix {mangled} now") == "please fix <lp0/> now"


def test_normalizer_leaves_correct_placeholders_untouched():
    assert normalize_placeholders("fix <lp0/> and <lp12/>") == "fix <lp0/> and <lp12/>"


def test_restore_after_normalizing_mangled_output():
    m = mask("`foo()` 를 고쳐")
    mangled = normalize_placeholders(m.masked.replace("<lp0/>", "[LP0]"))
    assert "foo()" in restore(mangled, m.placeholders)


def test_placeholder_ids_are_stable_and_sequential():
    m = mask("`a` 와 `b` 와 `c`")
    assert sorted(m.placeholders) == ["<lp0/>", "<lp1/>", "<lp2/>"]
