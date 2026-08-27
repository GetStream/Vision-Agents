from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="PlacedCall")


@_attrs_define
class PlacedCall:
    """
    Attributes:
        vendor_call_id (str):
        status (str): The vendor's own word for where the call is, e.g. "queued".
        vendor (str | Unset): Who is placing the call.
        call_id (str | Unset): The Stream call the answered leg is routed into. An agent that is not in it hears nothing
            when the person picks up.
        call_type (str | Unset):
    """

    vendor_call_id: str
    status: str
    vendor: str | Unset = UNSET
    call_id: str | Unset = UNSET
    call_type: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        vendor_call_id = self.vendor_call_id

        status = self.status

        vendor = self.vendor

        call_id = self.call_id

        call_type = self.call_type

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "vendor_call_id": vendor_call_id,
                "status": status,
            }
        )
        if vendor is not UNSET:
            field_dict["vendor"] = vendor
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if call_type is not UNSET:
            field_dict["call_type"] = call_type

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        vendor_call_id = d.pop("vendor_call_id")

        status = d.pop("status")

        vendor = d.pop("vendor", UNSET)

        call_id = d.pop("call_id", UNSET)

        call_type = d.pop("call_type", UNSET)

        placed_call = cls(
            vendor_call_id=vendor_call_id,
            status=status,
            vendor=vendor,
            call_id=call_id,
            call_type=call_type,
        )

        placed_call.additional_properties = d
        return placed_call

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
