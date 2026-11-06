from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.contact_input_variables import ContactInputVariables


T = TypeVar("T", bound="ContactInput")


@_attrs_define
class ContactInput:
    """
    Attributes:
        to_number (str): With or without a country code.
        instructions (str | Unset): What to say to this person, added to whatever the config already says. Up to 2,000
            characters; longer fails the whole request with 400.
        time_zone (str | Unset): An IANA zone, which a campaign reading its hours where each contact is goes by before
            the number's.
        variables (ContactInputVariables | Unset): Filled into the agent's {{placeholders}}. Names are lowercase
            letters, digits and underscores, starting with a letter; up to 50, each value up to 1,000 characters on one
            line.
    """

    to_number: str
    instructions: str | Unset = UNSET
    time_zone: str | Unset = UNSET
    variables: ContactInputVariables | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        to_number = self.to_number

        instructions = self.instructions

        time_zone = self.time_zone

        variables: dict[str, Any] | Unset = UNSET
        if not isinstance(self.variables, Unset):
            variables = self.variables.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "to_number": to_number,
            }
        )
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if time_zone is not UNSET:
            field_dict["time_zone"] = time_zone
        if variables is not UNSET:
            field_dict["variables"] = variables

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.contact_input_variables import ContactInputVariables

        d = dict(src_dict)
        to_number = d.pop("to_number")

        instructions = d.pop("instructions", UNSET)

        time_zone = d.pop("time_zone", UNSET)

        _variables = d.pop("variables", UNSET)
        variables: ContactInputVariables | Unset
        if isinstance(_variables, Unset):
            variables = UNSET
        else:
            variables = ContactInputVariables.from_dict(_variables)

        contact_input = cls(
            to_number=to_number,
            instructions=instructions,
            time_zone=time_zone,
            variables=variables,
        )

        contact_input.additional_properties = d
        return contact_input

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
