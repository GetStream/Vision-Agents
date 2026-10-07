from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="InputParts")


@_attrs_define
class InputParts:
    """What prompts were made of, in tokens. Estimated from each request and scaled to what the provider counted, so the
    parts sum to the input tokens and only the split between them is a guess. Requests recorded before the split was
    kept read zero throughout.

        Attributes:
            images (int): Pictures, attached or returned by a tool.
            instructions (int): The system prompt: the agent's instructions, skills and plugin guidance.
            messages (int): The conversation's words, from either side.
            tool_definitions (int): The tools the model was offered: their names, descriptions and schemas.
            tool_use (int): The tools the model called, and what they returned.
            video (int): Frames of a video, from the call's camera or an attached clip.
    """

    images: int
    instructions: int
    messages: int
    tool_definitions: int
    tool_use: int
    video: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        images = self.images

        instructions = self.instructions

        messages = self.messages

        tool_definitions = self.tool_definitions

        tool_use = self.tool_use

        video = self.video

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "images": images,
                "instructions": instructions,
                "messages": messages,
                "tool_definitions": tool_definitions,
                "tool_use": tool_use,
                "video": video,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        images = d.pop("images")

        instructions = d.pop("instructions")

        messages = d.pop("messages")

        tool_definitions = d.pop("tool_definitions")

        tool_use = d.pop("tool_use")

        video = d.pop("video")

        input_parts = cls(
            images=images,
            instructions=instructions,
            messages=messages,
            tool_definitions=tool_definitions,
            tool_use=tool_use,
            video=video,
        )

        input_parts.additional_properties = d
        return input_parts

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
