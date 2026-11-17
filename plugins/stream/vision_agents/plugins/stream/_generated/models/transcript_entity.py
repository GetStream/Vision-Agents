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
        text (str):
        type_ (str):  Example: person.
        end_ms (int | Unset):
        start_ms (int | Unset):
    """

    text: str
    type_: str
    end_ms: int | Unset = UNSET
    start_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        type_ = self.type_

        end_ms = self.end_ms

        start_ms = self.start_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "text": text,
                "type": type_,
            }
        )
        if end_ms is not UNSET:
            field_dict["end_ms"] = end_ms
        if start_ms is not UNSET:
            field_dict["start_ms"] = start_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        text = d.pop("text")

        type_ = d.pop("type")

        end_ms = d.pop("end_ms", UNSET)

        start_ms = d.pop("start_ms", UNSET)

        transcript_entity = cls(
            text=text,
            type_=type_,
            end_ms=end_ms,
            start_ms=start_ms,
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
