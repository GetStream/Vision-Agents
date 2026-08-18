from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.phone_capability import PhoneCapability
from ..types import UNSET, Unset

T = TypeVar("T", bound="AvailableNumber")


@_attrs_define
class AvailableNumber:
    """
    Attributes:
        e164 (str):  Example: +15125551234.
        country (str):
        capabilities (list[PhoneCapability]):
        region (str | Unset):
        locality (str | Unset):
        monthly_cost_micros (int | Unset): Millionths of a dollar per month, zero when the vendor does not quote one.
    """

    e164: str
    country: str
    capabilities: list[PhoneCapability]
    region: str | Unset = UNSET
    locality: str | Unset = UNSET
    monthly_cost_micros: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        e164 = self.e164

        country = self.country

        capabilities = []
        for capabilities_item_data in self.capabilities:
            capabilities_item = capabilities_item_data.value
            capabilities.append(capabilities_item)

        region = self.region

        locality = self.locality

        monthly_cost_micros = self.monthly_cost_micros

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "e164": e164,
                "country": country,
                "capabilities": capabilities,
            }
        )
        if region is not UNSET:
            field_dict["region"] = region
        if locality is not UNSET:
            field_dict["locality"] = locality
        if monthly_cost_micros is not UNSET:
            field_dict["monthly_cost_micros"] = monthly_cost_micros

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        e164 = d.pop("e164")

        country = d.pop("country")

        capabilities = []
        _capabilities = d.pop("capabilities")
        for capabilities_item_data in _capabilities:
            capabilities_item = PhoneCapability(capabilities_item_data)

            capabilities.append(capabilities_item)

        region = d.pop("region", UNSET)

        locality = d.pop("locality", UNSET)

        monthly_cost_micros = d.pop("monthly_cost_micros", UNSET)

        available_number = cls(
            e164=e164,
            country=country,
            capabilities=capabilities,
            region=region,
            locality=locality,
            monthly_cost_micros=monthly_cost_micros,
        )

        available_number.additional_properties = d
        return available_number

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
