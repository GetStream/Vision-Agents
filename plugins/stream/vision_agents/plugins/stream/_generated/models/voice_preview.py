from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="VoicePreview")


@_attrs_define
class VoicePreview:
    """
    Attributes:
        provider (str):
        content_type (str):  Example: audio/mpeg.
        audio (str): The spoken line, base64 encoded.
    """

    provider: str
    content_type: str
    audio: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider

        content_type = self.content_type

        audio = self.audio

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "provider": provider,
                "content_type": content_type,
                "audio": audio,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        provider = d.pop("provider")

        content_type = d.pop("content_type")

        audio = d.pop("audio")

        voice_preview = cls(
            provider=provider,
            content_type=content_type,
            audio=audio,
        )

        voice_preview.additional_properties = d
        return voice_preview

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
