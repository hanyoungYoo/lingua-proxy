"""Codecs locate translatable text without reshaping the request.

The hard requirement is surgical edits: one string replaced at one JSON
pointer. Block counts, ordering, sibling keys like cache_control, and any
field we do not understand must come out exactly as they went in.
"""

from __future__ import annotations

import copy

from lingua_proxy.codecs import AnthropicMessagesCodec, OpenAIChatCodec

anthropic = AnthropicMessagesCodec()
openai = OpenAIChatCodec()


# -- Anthropic ----------------------------------------------------------


def test_anthropic_extracts_string_content():
    body = {"messages": [{"role": "user", "content": "안녕하세요"}]}
    refs = anthropic.user_refs(body)

    assert [r.text for r in refs] == ["안녕하세요"]


def test_anthropic_extracts_text_blocks():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "첫 번째"},
                    {"type": "text", "text": "두 번째"},
                ],
            }
        ]
    }
    assert [r.text for r in anthropic.user_refs(body)] == ["첫 번째", "두 번째"]


def test_anthropic_skips_images_documents_and_tool_results():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "xxx"}},
                    {"type": "document", "source": {"type": "base64", "data": "yyy"}},
                    {"type": "tool_result", "tool_use_id": "t1", "content": "파일 내용"},
                    {"type": "text", "text": "이것만 번역"},
                ],
            }
        ]
    }
    assert [r.text for r in anthropic.user_refs(body)] == ["이것만 번역"]


def test_anthropic_skips_tool_result_with_block_content():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "command output"}],
                    }
                ],
            }
        ]
    }
    assert anthropic.user_refs(body) == []


def test_anthropic_translates_text_that_follows_tool_results():
    """The user typing while a tool ran is still a prompt."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "output"},
                    {"type": "text", "text": "이제 이걸 고쳐줘"},
                ],
            }
        ]
    }
    assert [r.text for r in anthropic.user_refs(body)] == ["이제 이걸 고쳐줘"]


def test_anthropic_only_reads_the_latest_user_turn():
    body = {
        "messages": [
            {"role": "user", "content": "예전 질문"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "새 질문"},
        ]
    }
    assert [r.text for r in anthropic.user_refs(body)] == ["새 질문"]


def test_anthropic_system_prompt_is_never_extracted():
    body = {
        "system": [{"type": "text", "text": "You are helpful."}],
        "messages": [{"role": "user", "content": "안녕"}],
    }
    assert [r.text for r in anthropic.user_refs(body)] == ["안녕"]


def test_anthropic_apply_preserves_structure_and_siblings():
    body = {
        "model": "m",
        "metadata": {"user_id": "u1"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "안녕하세요",
                        "cache_control": {"type": "ephemeral"},
                        "unknown_future_field": 7,
                    }
                ],
            }
        ],
    }
    original = copy.deepcopy(body)
    (ref,) = anthropic.user_refs(body)
    updated = anthropic.apply(body, ref, "Hello")

    block = updated["messages"][0]["content"][0]
    assert block["text"] == "Hello"
    assert block["cache_control"] == {"type": "ephemeral"}
    assert block["unknown_future_field"] == 7
    assert updated["metadata"] == {"user_id": "u1"}
    assert body == original, "apply must not mutate the caller's body"


def test_anthropic_apply_keeps_string_content_a_string():
    body = {"messages": [{"role": "user", "content": "안녕하세요"}]}
    (ref,) = anthropic.user_refs(body)
    updated = anthropic.apply(body, ref, "Hello")

    assert updated["messages"][0]["content"] == "Hello"
    assert isinstance(updated["messages"][0]["content"], str)


def test_anthropic_assistant_refs_find_history_replies():
    body = {
        "messages": [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": [{"type": "text", "text": "답변입니다"}]},
            {"role": "user", "content": "다음 질문"},
        ]
    }
    assert [r.text for r in anthropic.assistant_refs(body)] == ["답변입니다"]


def test_anthropic_assistant_refs_skip_tool_use_and_thinking():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
                    {"type": "text", "text": "결과입니다"},
                ],
            },
            {"role": "user", "content": "고마워"},
        ]
    }
    assert [r.text for r in anthropic.assistant_refs(body)] == ["결과입니다"]


def test_anthropic_response_refs_and_usage():
    resp = {
        "content": [
            {"type": "text", "text": "Here you go"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
        },
    }
    assert [r.text for r in anthropic.response_refs(resp)] == ["Here you go"]

    usage = anthropic.usage(resp)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cache_read_input_tokens == 20
    assert usage.cache_creation_input_tokens == 10


def test_anthropic_usage_defaults_to_zero_when_absent():
    usage = anthropic.usage({"content": []})
    assert usage.input_tokens == 0 and usage.output_tokens == 0


def test_anthropic_no_user_message_yields_no_refs():
    assert anthropic.user_refs({"messages": [{"role": "assistant", "content": "hi"}]}) == []


def test_anthropic_latest_turn_not_user_yields_no_refs():
    body = {
        "messages": [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "prefill"},
        ]
    }
    assert anthropic.user_refs(body) == []


def test_anthropic_malformed_body_is_safe():
    assert anthropic.user_refs({}) == []
    assert anthropic.user_refs({"messages": "not a list"}) == []
    assert anthropic.user_refs({"messages": [{"role": "user"}]}) == []


def test_anthropic_model_and_stream_accessors():
    body = {"model": "claude-sonnet-4-6", "stream": True, "messages": []}
    assert anthropic.model(body) == "claude-sonnet-4-6"
    assert anthropic.is_stream(body) is True
    assert anthropic.is_stream({"messages": []}) is False


# -- OpenAI -------------------------------------------------------------


def test_openai_extracts_string_content():
    body = {"messages": [{"role": "user", "content": "안녕하세요"}]}
    assert [r.text for r in openai.user_refs(body)] == ["안녕하세요"]


def test_openai_extracts_text_parts_and_skips_images():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 이미지를 설명해"},
                    {"type": "image_url", "image_url": {"url": "https://gw.example/a.png"}},
                ],
            }
        ]
    }
    assert [r.text for r in openai.user_refs(body)] == ["이 이미지를 설명해"]


def test_openai_skips_system_developer_and_tool_roles():
    body = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "developer", "content": "Be terse"},
            {"role": "tool", "tool_call_id": "c1", "content": "output"},
            {"role": "user", "content": "번역할 것"},
        ]
    }
    assert [r.text for r in openai.user_refs(body)] == ["번역할 것"]


def test_openai_assistant_with_null_content_and_tool_calls_is_safe():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "user", "content": "계속해"},
        ]
    }
    assert [r.text for r in openai.user_refs(body)] == ["계속해"]
    assert openai.assistant_refs(body) == []


def test_openai_assistant_refs_find_history_replies():
    body = {
        "messages": [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답변입니다"},
            {"role": "user", "content": "다음"},
        ]
    }
    assert [r.text for r in openai.assistant_refs(body)] == ["답변입니다"]


def test_openai_apply_preserves_structure():
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "안녕", "extra": 1}]}]
    }
    (ref,) = openai.user_refs(body)
    updated = openai.apply(body, ref, "Hello")

    part = updated["messages"][0]["content"][0]
    assert part["text"] == "Hello"
    assert part["extra"] == 1


def test_openai_response_refs_and_usage():
    resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Here you go"}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 12},
    }
    assert [r.text for r in openai.response_refs(resp)] == ["Here you go"]

    usage = openai.usage(resp)
    assert usage.input_tokens == 30
    assert usage.output_tokens == 12


def test_openai_response_with_null_content_yields_no_refs():
    resp = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    assert openai.response_refs(resp) == []


def test_history_user_refs_excludes_the_latest_turn():
    """Earlier turns are memo-reproduced; the latest one is freshly translated."""
    body = {
        "messages": [
            {"role": "user", "content": "예전 질문"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "새 질문"},
        ]
    }
    assert [r.text for r in anthropic.history_user_refs(body)] == ["예전 질문"]
    assert [r.text for r in anthropic.user_refs(body)] == ["새 질문"]


def test_history_user_refs_is_empty_for_a_first_turn():
    body = {"messages": [{"role": "user", "content": "첫 질문"}]}
    assert anthropic.history_user_refs(body) == []


def test_openai_history_user_refs_excludes_the_latest_turn():
    body = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "예전 질문"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "새 질문"},
        ]
    }
    assert [r.text for r in openai.history_user_refs(body)] == ["예전 질문"]
