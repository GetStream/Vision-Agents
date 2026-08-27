from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.phone_capability import PhoneCapability
from ..models.phone_operation import PhoneOperation
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
        operations (list[PhoneOperation] | Unset): What this service can do at the vendor. Eight vendors buy numbers and
            two of those also bridge calls, so a number is not bought from a vendor that cannot answer on it by accident.
        missing_credentials (list[str] | Unset): The environment variables the vendor needs and does not have.
    """

    vendor: str
    implemented: bool
    ready: bool
    capabilities: list[PhoneCapability]
    operations: list[PhoneOperation] | Unset = UNSET
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

        operations: list[str] | Unset = UNSET
        if not isinstance(self.operations, Unset):
            operations = []
            for operations_item_data in self.operations:
                operations_item = operations_item_data.value
                operations.append(operations_item)

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
        if operations is not UNSET:
            field_dict["operations"] = operations
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

        _operations = d.pop("operations", UNSET)
        operations: list[PhoneOperation] | Unset = UNSET
        if _operations is not UNSET:
            operations = []
            for operations_item_data in _operations:
                operations_item = PhoneOperation(operations_item_data)

                operations.append(operations_item)

        missing_credentials = cast(list[str], d.pop("missing_credentials", UNSET))

        phone_vendor = cls(
            vendor=vendor,
            implemented=implemented,
            ready=ready,
            capabilities=capabilities,
            operations=operations,
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
