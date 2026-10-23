from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ResumeHistory")


@_attrs_define
class ResumeHistory:
    """How a conversation is handed back to the model when a session carries it on.

    Attributes:
        include_tools (bool | Unset): Hand each reply back with the tool calls it made and what they returned, as they
            were recorded, so a resumed agent knows what it already looked up and does not run its tools again. Over
            max_characters the oldest results are replaced with [result trimmed] before any turn is left out, and the
            conversation's first message is kept. A reply whose calls were not recorded, have been removed with their
            session or redacted under a retention setting is handed back as text. Results are replayed as recorded:
            credentials redacted and any result over 8 KB cut to its first 2048 characters. Off by default.
        max_characters (int | Unset): How many characters are handed back, tool calls and results included: about four
            to a token. The whole of it is sent, and billed, with the first reply after the conversation is resumed, and is
            cached from then on by models that cache, so keep it well inside the model's context window. Omitted is 60000.
        max_messages (int | Unset): How many turns are handed back, newest first. Tool calls and their results are not
            turns. Omitted is 100.
    """

    include_tools: bool | Unset = UNSET
    max_characters: int | Unset = UNSET
    max_messages: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        include_tools = self.include_tools

        max_characters = self.max_characters

        max_messages = self.max_messages

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if include_tools is not UNSET:
            field_dict["include_tools"] = include_tools
        if max_characters is not UNSET:
            field_dict["max_characters"] = max_characters
        if max_messages is not UNSET:
            field_dict["max_messages"] = max_messages

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        include_tools = d.pop("include_tools", UNSET)

        max_characters = d.pop("max_characters", UNSET)

        max_messages = d.pop("max_messages", UNSET)

        resume_history = cls(
            include_tools=include_tools,
            max_characters=max_characters,
            max_messages=max_messages,
        )

        resume_history.additional_properties = d
        return resume_history

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
