from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.available_number import AvailableNumber
    from ..models.skipped_vendor import SkippedVendor


T = TypeVar("T", bound="NumberSearchResult")


@_attrs_define
class NumberSearchResult:
    """
    Attributes:
        numbers (list[AvailableNumber]): What the vendors are offering, cheapest first.
        skipped (list[SkippedVendor]): Vendors that were not part of the answer. A search that reached two of eight
            vendors found what two vendors had, and deciding whether to buy needs to know which.
    """

    numbers: list[AvailableNumber]
    skipped: list[SkippedVendor]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        numbers = []
        for numbers_item_data in self.numbers:
            numbers_item = numbers_item_data.to_dict()
            numbers.append(numbers_item)

        skipped = []
        for skipped_item_data in self.skipped:
            skipped_item = skipped_item_data.to_dict()
            skipped.append(skipped_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "numbers": numbers,
                "skipped": skipped,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.available_number import AvailableNumber
        from ..models.skipped_vendor import SkippedVendor

        d = dict(src_dict)
        numbers = []
        _numbers = d.pop("numbers")
        for numbers_item_data in _numbers:
            numbers_item = AvailableNumber.from_dict(numbers_item_data)

            numbers.append(numbers_item)

        skipped = []
        _skipped = d.pop("skipped")
        for skipped_item_data in _skipped:
            skipped_item = SkippedVendor.from_dict(skipped_item_data)

            skipped.append(skipped_item)

        number_search_result = cls(
            numbers=numbers,
            skipped=skipped,
        )

        number_search_result.additional_properties = d
        return number_search_result

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
