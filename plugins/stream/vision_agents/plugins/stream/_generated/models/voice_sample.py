from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="VoiceSample")


@_attrs_define
class VoiceSample:
    """
    Attributes:
        created_at (datetime.datetime):
        id (str):
        bytes_ (int | Unset):
        content_type (str | Unset):
        filename (str | Unset):
        transcript (str | Unset):
    """

    created_at: datetime.datetime
    id: str
    bytes_: int | Unset = UNSET
    content_type: str | Unset = UNSET
    filename: str | Unset = UNSET
    transcript: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        id = self.id

        bytes_ = self.bytes_

        content_type = self.content_type

        filename = self.filename

        transcript = self.transcript

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "id": id,
            }
        )
        if bytes_ is not UNSET:
            field_dict["bytes"] = bytes_
        if content_type is not UNSET:
            field_dict["content_type"] = content_type
        if filename is not UNSET:
            field_dict["filename"] = filename
        if transcript is not UNSET:
            field_dict["transcript"] = transcript

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        bytes_ = d.pop("bytes", UNSET)

        content_type = d.pop("content_type", UNSET)

        filename = d.pop("filename", UNSET)

        transcript = d.pop("transcript", UNSET)

        voice_sample = cls(
            created_at=created_at,
            id=id,
            bytes_=bytes_,
            content_type=content_type,
            filename=filename,
            transcript=transcript,
        )

        voice_sample.additional_properties = d
        return voice_sample

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
