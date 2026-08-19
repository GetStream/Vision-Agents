from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ContactsRequestContactsItem")


@_attrs_define
class ContactsRequestContactsItem:
    """
    Attributes:
        to_number (str):
        instructions (str | Unset): What to say to this person, added to whatever the config already says.
    """

    to_number: str
    instructions: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        to_number = self.to_number

        instructions = self.instructions

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "to_number": to_number,
            }
        )
        if instructions is not UNSET:
            field_dict["instructions"] = instructions

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        to_number = d.pop("to_number")

        instructions = d.pop("instructions", UNSET)

        contacts_request_contacts_item = cls(
            to_number=to_number,
            instructions=instructions,
        )

        contacts_request_contacts_item.additional_properties = d
        return contacts_request_contacts_item

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
