from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.phone_capability import PhoneCapability
from ..models.phone_number_type import PhoneNumberType
from ..types import UNSET, Unset

T = TypeVar("T", bound="AvailableNumber")


@_attrs_define
class AvailableNumber:
    """
    Attributes:
        capabilities (list[PhoneCapability]):
        country (str):
        e164 (str):  Example: +15125551234.
        vendor (str): Who is offering it, which is also who to buy it from. Example: telnyx.
        locality (str | Unset):
        monthly_cost_micros (int | Unset): Millionths of a dollar per month, zero when the vendor does not quote one.
        number_type (PhoneNumberType | Unset): What kind of number it is, which decides who pays for the call.
        region (str | Unset):
    """

    capabilities: list[PhoneCapability]
    country: str
    e164: str
    vendor: str
    locality: str | Unset = UNSET
    monthly_cost_micros: int | Unset = UNSET
    number_type: PhoneNumberType | Unset = UNSET
    region: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        capabilities = []
        for capabilities_item_data in self.capabilities:
            capabilities_item = capabilities_item_data.value
            capabilities.append(capabilities_item)

        country = self.country

        e164 = self.e164

        vendor = self.vendor

        locality = self.locality

        monthly_cost_micros = self.monthly_cost_micros

        number_type: str | Unset = UNSET
        if not isinstance(self.number_type, Unset):
            number_type = self.number_type.value

        region = self.region

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "capabilities": capabilities,
                "country": country,
                "e164": e164,
                "vendor": vendor,
            }
        )
        if locality is not UNSET:
            field_dict["locality"] = locality
        if monthly_cost_micros is not UNSET:
            field_dict["monthly_cost_micros"] = monthly_cost_micros
        if number_type is not UNSET:
            field_dict["number_type"] = number_type
        if region is not UNSET:
            field_dict["region"] = region

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        capabilities = []
        _capabilities = d.pop("capabilities")
        for capabilities_item_data in _capabilities:
            capabilities_item = PhoneCapability(capabilities_item_data)

            capabilities.append(capabilities_item)

        country = d.pop("country")

        e164 = d.pop("e164")

        vendor = d.pop("vendor")

        locality = d.pop("locality", UNSET)

        monthly_cost_micros = d.pop("monthly_cost_micros", UNSET)

        _number_type = d.pop("number_type", UNSET)
        number_type: PhoneNumberType | Unset
        if isinstance(_number_type, Unset):
            number_type = UNSET
        else:
            number_type = PhoneNumberType(_number_type)

        region = d.pop("region", UNSET)

        available_number = cls(
            capabilities=capabilities,
            country=country,
            e164=e164,
            vendor=vendor,
            locality=locality,
            monthly_cost_micros=monthly_cost_micros,
            number_type=number_type,
            region=region,
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
