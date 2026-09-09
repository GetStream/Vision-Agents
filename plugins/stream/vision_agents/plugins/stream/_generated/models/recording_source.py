from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="RecordingSource")


@_attrs_define
class RecordingSource:
    """Where the audio to work on comes from. A URL is what every vendor's batch API takes and what anything longer than a
    clip should use; inline bytes save a caller with a short local file from having to host it somewhere first.

        Attributes:
            url (str | Unset): A fetchable audio or video file.
            audio (str | Unset): The file itself, base64. For clips - a long recording belongs behind a URL.
    """

    url: str | Unset = UNSET
    audio: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        url = self.url

        audio = self.audio

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if url is not UNSET:
            field_dict["url"] = url
        if audio is not UNSET:
            field_dict["audio"] = audio

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        url = d.pop("url", UNSET)

        audio = d.pop("audio", UNSET)

        recording_source = cls(
            url=url,
            audio=audio,
        )

        recording_source.additional_properties = d
        return recording_source

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
