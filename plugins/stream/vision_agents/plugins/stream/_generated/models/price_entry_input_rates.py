from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.price_rate import PriceRate


T = TypeVar("T", bound="PriceEntryInputRates")


@_attrs_define
class PriceEntryInputRates:
    """The vendor's cost, by meter, kept for reference: input_tokens, cached_input_tokens, cache_write_input_tokens,
    output_tokens, characters, audio_ms, images, requests, message_segments, call_seconds and the rest pricing knows.
    Required unless the entry is free or withdrawn. An unknown meter, and a rate of more than $1000 a unit, are refused.

    """

    additional_properties: dict[str, PriceRate] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_rate import PriceRate

        d = dict(src_dict)
        price_entry_input_rates = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = PriceRate.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        price_entry_input_rates.additional_properties = additional_properties
        return price_entry_input_rates

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> PriceRate:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: PriceRate) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
