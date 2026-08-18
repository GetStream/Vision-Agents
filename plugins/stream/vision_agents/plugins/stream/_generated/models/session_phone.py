from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SessionPhone")


@_attrs_define
class SessionPhone:
    """The number the session acts from, which is what turns transferring on.

    Attributes:
        number (str): One of the customer's own numbers, written as +15551234567.
        vendor (str | Unset): Who carries an outbound leg.
        vendor_call_id (str | Unset): The outbound leg, set for a call the agent placed. Without one the agent has no
            keypad to press at.
    """

    number: str
    vendor: str | Unset = UNSET
    vendor_call_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        number = self.number

        vendor = self.vendor

        vendor_call_id = self.vendor_call_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "number": number,
            }
        )
        if vendor is not UNSET:
            field_dict["vendor"] = vendor
        if vendor_call_id is not UNSET:
            field_dict["vendor_call_id"] = vendor_call_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        number = d.pop("number")

        vendor = d.pop("vendor", UNSET)

        vendor_call_id = d.pop("vendor_call_id", UNSET)

        session_phone = cls(
            number=number,
            vendor=vendor,
            vendor_call_id=vendor_call_id,
        )

        session_phone.additional_properties = d
        return session_phone

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
