"""Wire-format adapters.

Two request formats reach this proxy: the Anthropic Messages API (what Claude
Code speaks) and the OpenAI chat-completions API. The pipeline is written once
against the :class:`Codec` interface; these classes know the JSON shapes.

The central discipline is **surgical edits**. A codec never rebuilds a request.
It reports a :class:`Ref` -- a path to exactly one string -- and ``apply``
replaces that one string on a deep copy. Block counts, ordering, sibling keys
such as ``cache_control``, and fields we have never heard of survive untouched.
Flattening block-form content would silently break prompt caching, so it is
never done.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

#: Content block types that must never be translated. Tool results are file
#: contents and command output; images and documents are not text; thinking is
#: the model's own reasoning and is signature-bound.
_SKIP_BLOCK_TYPES = frozenset(
    {"tool_result", "tool_use", "image", "document", "thinking", "redacted_thinking"}
)

#: Roles whose content is never the user's own prose.
_OPENAI_SKIP_ROLES = frozenset({"system", "developer", "tool", "function"})


@dataclass(frozen=True)
class Ref:
    """A pointer to one translatable string inside a request or response."""

    path: tuple[str | int, ...]
    text: str
    role: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    """Token usage, normalized across wire formats."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


def _set_in(body: dict, path: tuple[str | int, ...], value: str) -> dict:
    """Return a copy of ``body`` with one leaf replaced."""
    updated = copy.deepcopy(body)
    cursor: Any = updated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return updated


class Codec:
    """Interface the pipeline programs against."""

    def user_refs(self, body: dict) -> list[Ref]:
        raise NotImplementedError

    def history_user_refs(self, body: dict) -> list[Ref]:
        """User text from every turn *except* the latest.

        These are reproduced from the memo, never re-translated, so the bytes
        the upstream sees for earlier turns never change.
        """
        raise NotImplementedError

    def assistant_refs(self, body: dict) -> list[Ref]:
        raise NotImplementedError

    def response_refs(self, response: dict) -> list[Ref]:
        raise NotImplementedError

    def apply(self, body: dict, ref: Ref, text: str) -> dict:
        return _set_in(body, ref.path, text)

    def usage(self, response: dict) -> Usage:
        raise NotImplementedError

    def model(self, body: dict) -> str:
        value = body.get("model") if isinstance(body, dict) else None
        return value if isinstance(value, str) else ""

    def is_stream(self, body: dict) -> bool:
        return bool(isinstance(body, dict) and body.get("stream"))

    def max_output_tokens(self, body: dict) -> int | None:
        """The reply ceiling this request asked for, if it named one."""
        if not isinstance(body, dict):
            return None
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = body.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return None


def _messages(body: dict) -> list[dict]:
    if not isinstance(body, dict):
        return []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


class AnthropicMessagesCodec(Codec):
    """``POST /v1/messages`` -- the format Claude Code speaks."""

    def _text_refs_for_message(self, index: int, message: dict) -> list[Ref]:
        role = message.get("role", "")
        content = message.get("content")

        if isinstance(content, str):
            if not content.strip():
                return []
            return [Ref(path=("messages", index, "content"), text=content, role=role)]

        if not isinstance(content, list):
            return []

        refs = []
        for position, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            refs.append(
                Ref(
                    path=("messages", index, "content", position, "text"),
                    text=text,
                    role=role,
                )
            )
        return refs

    def user_refs(self, body: dict) -> list[Ref]:
        """Translatable text in the latest user turn only.

        Earlier turns are handled by the memo: re-translating them would
        change bytes the upstream has already cached.
        """
        messages = _messages(body)
        if not messages:
            return []

        index = len(messages) - 1
        last = messages[index]
        if last.get("role") != "user":
            return []
        return self._text_refs_for_message(index, last)

    def history_user_refs(self, body: dict) -> list[Ref]:
        messages = _messages(body)
        refs = []
        for index, message in enumerate(messages[:-1]):
            if message.get("role") != "user":
                continue
            refs.extend(self._text_refs_for_message(index, message))
        return refs

    def assistant_refs(self, body: dict) -> list[Ref]:
        refs = []
        for index, message in enumerate(_messages(body)):
            if message.get("role") != "assistant":
                continue
            refs.extend(self._text_refs_for_message(index, message))
        return refs

    def response_refs(self, response: dict) -> list[Ref]:
        content = response.get("content") if isinstance(response, dict) else None
        if not isinstance(content, list):
            return []

        refs = []
        for position, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            refs.append(Ref(path=("content", position, "text"), text=text, role="assistant"))
        return refs

    def usage(self, response: dict) -> Usage:
        raw = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(raw, dict):
            return Usage()
        return Usage(
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
            cache_read_input_tokens=int(raw.get("cache_read_input_tokens") or 0),
            cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens") or 0),
        )


class OpenAIChatCodec(Codec):
    """``POST /v1/chat/completions`` -- the OpenAI-compatible format."""

    def _text_refs_for_message(self, index: int, message: dict) -> list[Ref]:
        role = message.get("role", "")
        content = message.get("content")

        if isinstance(content, str):
            if not content.strip():
                return []
            return [Ref(path=("messages", index, "content"), text=content, role=role)]

        if not isinstance(content, list):
            # Assistant turns carrying only tool_calls have content: null.
            return []

        refs = []
        for position, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            refs.append(
                Ref(
                    path=("messages", index, "content", position, "text"),
                    text=text,
                    role=role,
                )
            )
        return refs

    def user_refs(self, body: dict) -> list[Ref]:
        messages = _messages(body)
        if not messages:
            return []

        index = len(messages) - 1
        last = messages[index]
        if last.get("role") != "user" or last.get("role") in _OPENAI_SKIP_ROLES:
            return []
        return self._text_refs_for_message(index, last)

    def history_user_refs(self, body: dict) -> list[Ref]:
        messages = _messages(body)
        refs = []
        for index, message in enumerate(messages[:-1]):
            if message.get("role") != "user":
                continue
            refs.extend(self._text_refs_for_message(index, message))
        return refs

    def assistant_refs(self, body: dict) -> list[Ref]:
        refs = []
        for index, message in enumerate(_messages(body)):
            if message.get("role") != "assistant":
                continue
            refs.extend(self._text_refs_for_message(index, message))
        return refs

    def response_refs(self, response: dict) -> list[Ref]:
        choices = response.get("choices") if isinstance(response, dict) else None
        if not isinstance(choices, list):
            return []

        refs = []
        for index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            refs.append(
                Ref(
                    path=("choices", index, "message", "content"),
                    text=content,
                    role="assistant",
                )
            )
        return refs

    def usage(self, response: dict) -> Usage:
        raw = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(raw, dict):
            return Usage()
        details = raw.get("prompt_tokens_details")
        cached = 0
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens") or 0)
        return Usage(
            input_tokens=int(raw.get("prompt_tokens") or 0),
            output_tokens=int(raw.get("completion_tokens") or 0),
            cache_read_input_tokens=cached,
        )
