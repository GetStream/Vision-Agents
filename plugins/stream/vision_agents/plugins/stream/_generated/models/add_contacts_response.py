from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.refused_contact import RefusedContact


T = TypeVar("T", bound="AddContactsResponse")


@_attrs_define
class AddContactsResponse:
    """
    Attributes:
        added (int):
        refused (list[RefusedContact]): In the order sent.
    """

    added: int
    refused: list[RefusedContact]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        added = self.added

        refused = []
        for refused_item_data in self.refused:
            refused_item = refused_item_data.to_dict()
            refused.append(refused_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "added": added,
                "refused": refused,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.refused_contact import RefusedContact

        d = dict(src_dict)
        added = d.pop("added")

        refused = []
        _refused = d.pop("refused")
        for refused_item_data in _refused:
            refused_item = RefusedContact.from_dict(refused_item_data)

            refused.append(refused_item)

        add_contacts_response = cls(
            added=added,
            refused=refused,
        )

        add_contacts_response.additional_properties = d
        return add_contacts_response

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
