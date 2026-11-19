from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.contact_update_request_variables import ContactUpdateRequestVariables


T = TypeVar("T", bound="ContactUpdateRequest")


@_attrs_define
class ContactUpdateRequest:
    """
    Attributes:
        instructions (str | Unset): What to say to this person. Empty clears it.
        time_zone (str | Unset): An IANA zone. Empty clears it.
        to_number (str | Unset): With or without a country code.
        variables (ContactUpdateRequestVariables | Unset): Replace all of the contact's variables. Same rules as on add.
    """

    instructions: str | Unset = UNSET
    time_zone: str | Unset = UNSET
    to_number: str | Unset = UNSET
    variables: ContactUpdateRequestVariables | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        instructions = self.instructions

        time_zone = self.time_zone

        to_number = self.to_number

        variables: dict[str, Any] | Unset = UNSET
        if not isinstance(self.variables, Unset):
            variables = self.variables.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if time_zone is not UNSET:
            field_dict["time_zone"] = time_zone
        if to_number is not UNSET:
            field_dict["to_number"] = to_number
        if variables is not UNSET:
            field_dict["variables"] = variables

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.contact_update_request_variables import (
            ContactUpdateRequestVariables,
        )

        d = dict(src_dict)
        instructions = d.pop("instructions", UNSET)

        time_zone = d.pop("time_zone", UNSET)

        to_number = d.pop("to_number", UNSET)

        _variables = d.pop("variables", UNSET)
        variables: ContactUpdateRequestVariables | Unset
        if isinstance(_variables, Unset):
            variables = UNSET
        else:
            variables = ContactUpdateRequestVariables.from_dict(_variables)

        contact_update_request = cls(
            instructions=instructions,
            time_zone=time_zone,
            to_number=to_number,
            variables=variables,
        )

        contact_update_request.additional_properties = d
        return contact_update_request

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
