from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="PricePreviewItemPriceFactors")


@_attrs_define
class PricePreviewItemPriceFactors:
    """By meter, the factor its price moves by. Absent for a meter with no price before or after."""

    additional_properties: dict[str, float] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        price_preview_item_price_factors = cls()

        price_preview_item_price_factors.additional_properties = d
        return price_preview_item_price_factors

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> float:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: float) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
