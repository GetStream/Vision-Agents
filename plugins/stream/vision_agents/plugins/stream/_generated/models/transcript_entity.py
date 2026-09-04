from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="TranscriptEntity")


@_attrs_define
class TranscriptEntity:
    """Something the recording named, for the providers that pick them out.

    Attributes:
        type_ (str):  Example: person.
        text (str):
        start_ms (int | Unset):
        end_ms (int | Unset):
    """

    type_: str
    text: str
    start_ms: int | Unset = UNSET
    end_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        text = self.text

        start_ms = self.start_ms

        end_ms = self.end_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
                "text": text,
            }
        )
        if start_ms is not UNSET:
            field_dict["start_ms"] = start_ms
        if end_ms is not UNSET:
            field_dict["end_ms"] = end_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        type_ = d.pop("type")

        text = d.pop("text")

        start_ms = d.pop("start_ms", UNSET)

        end_ms = d.pop("end_ms", UNSET)

        transcript_entity = cls(
            type_=type_,
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        transcript_entity.additional_properties = d
        return transcript_entity

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
