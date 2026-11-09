from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.contact_input import ContactInput


T = TypeVar("T", bound="AddContactsRequest")


@_attrs_define
class AddContactsRequest:
    """
    Attributes:
        contacts (list[ContactInput]): Up to 1,000.
        default_country (str | Unset): The region, as ISO 3166-1 alpha-2, a number without a country code is read in.
            Omitted is the campaign's default_country, else US.
    """

    contacts: list[ContactInput]
    default_country: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        contacts = []
        for contacts_item_data in self.contacts:
            contacts_item = contacts_item_data.to_dict()
            contacts.append(contacts_item)

        default_country = self.default_country

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "contacts": contacts,
            }
        )
        if default_country is not UNSET:
            field_dict["default_country"] = default_country

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.contact_input import ContactInput

        d = dict(src_dict)
        contacts = []
        _contacts = d.pop("contacts")
        for contacts_item_data in _contacts:
            contacts_item = ContactInput.from_dict(contacts_item_data)

            contacts.append(contacts_item)

        default_country = d.pop("default_country", UNSET)

        add_contacts_request = cls(
            contacts=contacts,
            default_country=default_country,
        )

        add_contacts_request.additional_properties = d
        return add_contacts_request

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
