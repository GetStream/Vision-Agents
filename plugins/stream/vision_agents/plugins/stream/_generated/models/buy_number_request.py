from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.buy_number_request_tags import BuyNumberRequestTags


T = TypeVar("T", bound="BuyNumberRequest")


@_attrs_define
class BuyNumberRequest:
    """
    Attributes:
        vendor (str):  Example: twilio.
        e164 (str):  Example: +15125551234.
        tags (BuyNumberRequestTags | Unset): Cost labels carried onto the purchase's request row.
    """

    vendor: str
    e164: str
    tags: BuyNumberRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        vendor = self.vendor

        e164 = self.e164

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "vendor": vendor,
                "e164": e164,
            }
        )
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.buy_number_request_tags import BuyNumberRequestTags

        d = dict(src_dict)
        vendor = d.pop("vendor")

        e164 = d.pop("e164")

        _tags = d.pop("tags", UNSET)
        tags: BuyNumberRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = BuyNumberRequestTags.from_dict(_tags)

        buy_number_request = cls(
            vendor=vendor,
            e164=e164,
            tags=tags,
        )

        buy_number_request.additional_properties = d
        return buy_number_request

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
