from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.duplicate_campaign_request_contacts import (
    DuplicateCampaignRequestContacts,
)

T = TypeVar("T", bound="DuplicateCampaignRequest")


@_attrs_define
class DuplicateCampaignRequest:
    """
    Attributes:
        contacts (DuplicateCampaignRequestContacts): none, all, or the contacts never rung, which a stopped or completed
            campaign alone knows.
    """

    contacts: DuplicateCampaignRequestContacts
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        contacts = self.contacts.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "contacts": contacts,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        contacts = DuplicateCampaignRequestContacts._decode(d.pop("contacts"))

        duplicate_campaign_request = cls(
            contacts=contacts,
        )

        duplicate_campaign_request.additional_properties = d
        return duplicate_campaign_request

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
