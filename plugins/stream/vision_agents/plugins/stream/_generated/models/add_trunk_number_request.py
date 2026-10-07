from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.add_trunk_number_request_tags import AddTrunkNumberRequestTags


T = TypeVar("T", bound="AddTrunkNumberRequest")


@_attrs_define
class AddTrunkNumberRequest:
    """
    Attributes:
        country (str | Unset): Required. ISO 3166-1 alpha-2 country code.
        e164 (str | Unset): Required. The number in +15551234567 form.
        tags (AddTrunkNumberRequestTags | Unset): The customer's own cost labels.
    """

    country: str | Unset = UNSET
    e164: str | Unset = UNSET
    tags: AddTrunkNumberRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        country = self.country

        e164 = self.e164

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if country is not UNSET:
            field_dict["country"] = country
        if e164 is not UNSET:
            field_dict["e164"] = e164
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.add_trunk_number_request_tags import (
            AddTrunkNumberRequestTags,
        )

        d = dict(src_dict)
        country = d.pop("country", UNSET)

        e164 = d.pop("e164", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: AddTrunkNumberRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = AddTrunkNumberRequestTags.from_dict(_tags)

        add_trunk_number_request = cls(
            country=country,
            e164=e164,
            tags=tags,
        )

        add_trunk_number_request.additional_properties = d
        return add_trunk_number_request

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
