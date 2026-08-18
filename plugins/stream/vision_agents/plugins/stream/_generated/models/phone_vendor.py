from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.phone_capability import PhoneCapability
from ..types import UNSET, Unset

T = TypeVar("T", bound="PhoneVendor")


@_attrs_define
class PhoneVendor:
    """
    Attributes:
        vendor (str):  Example: twilio.
        implemented (bool): Whether this service can actually work with the vendor.
        ready (bool): Implemented and holding every credential it needs.
        capabilities (list[PhoneCapability]):
        missing_credentials (list[str] | Unset): The environment variables the vendor needs and does not have.
    """

    vendor: str
    implemented: bool
    ready: bool
    capabilities: list[PhoneCapability]
    missing_credentials: list[str] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        vendor = self.vendor

        implemented = self.implemented

        ready = self.ready

        capabilities = []
        for capabilities_item_data in self.capabilities:
            capabilities_item = capabilities_item_data.value
            capabilities.append(capabilities_item)

        missing_credentials: list[str] | Unset = UNSET
        if not isinstance(self.missing_credentials, Unset):
            missing_credentials = self.missing_credentials

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "vendor": vendor,
                "implemented": implemented,
                "ready": ready,
                "capabilities": capabilities,
            }
        )
        if missing_credentials is not UNSET:
            field_dict["missing_credentials"] = missing_credentials

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        vendor = d.pop("vendor")

        implemented = d.pop("implemented")

        ready = d.pop("ready")

        capabilities = []
        _capabilities = d.pop("capabilities")
        for capabilities_item_data in _capabilities:
            capabilities_item = PhoneCapability(capabilities_item_data)

            capabilities.append(capabilities_item)

        missing_credentials = cast(list[str], d.pop("missing_credentials", UNSET))

        phone_vendor = cls(
            vendor=vendor,
            implemented=implemented,
            ready=ready,
            capabilities=capabilities,
            missing_credentials=missing_credentials,
        )

        phone_vendor.additional_properties = d
        return phone_vendor

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
