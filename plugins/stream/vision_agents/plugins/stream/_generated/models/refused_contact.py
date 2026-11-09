from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.refused_reason import RefusedReason

T = TypeVar("T", bound="RefusedContact")


@_attrs_define
class RefusedContact:
    """
    Attributes:
        index (int): Its place in contacts, from 0.
        reason (RefusedReason): Why a contact was not added: invalid_number is a number that cannot be read, duplicate
            one the campaign already holds or the request repeats, invalid_variable more than 50 variables, a variable with
            a bad name, or a value too long or over more than one line, invalid_time_zone a time zone that is not an IANA
            one.
        to_number (str): As it was sent.
    """

    index: int
    reason: RefusedReason
    to_number: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        index = self.index

        reason = self.reason.value

        to_number = self.to_number

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "index": index,
                "reason": reason,
                "to_number": to_number,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        index = d.pop("index")

        reason = RefusedReason._decode(d.pop("reason"))

        to_number = d.pop("to_number")

        refused_contact = cls(
            index=index,
            reason=reason,
            to_number=to_number,
        )

        refused_contact.additional_properties = d
        return refused_contact

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
