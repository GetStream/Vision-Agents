from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="VoiceSampleRequest")


@_attrs_define
class VoiceSampleRequest:
    """
    Attributes:
        audio (str): The recording, base64 encoded. A minute of clean speech is plenty.
        filename (str | Unset): What to call the file upstream. The extension is how a provider knows what it was given,
            so send one.
        content_type (str | Unset):
        transcript (str | Unset): What is said in the recording. Optional, and the providers that use one clone more
            faithfully with it.
    """

    audio: str
    filename: str | Unset = UNSET
    content_type: str | Unset = UNSET
    transcript: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        audio = self.audio

        filename = self.filename

        content_type = self.content_type

        transcript = self.transcript

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "audio": audio,
            }
        )
        if filename is not UNSET:
            field_dict["filename"] = filename
        if content_type is not UNSET:
            field_dict["content_type"] = content_type
        if transcript is not UNSET:
            field_dict["transcript"] = transcript

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        audio = d.pop("audio")

        filename = d.pop("filename", UNSET)

        content_type = d.pop("content_type", UNSET)

        transcript = d.pop("transcript", UNSET)

        voice_sample_request = cls(
            audio=audio,
            filename=filename,
            content_type=content_type,
            transcript=transcript,
        )

        voice_sample_request.additional_properties = d
        return voice_sample_request

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
