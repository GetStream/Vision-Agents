from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="PressDigitsRequest")


@_attrs_define
class PressDigitsRequest:
    """
    Attributes:
        vendor (str): Who is carrying the call, e.g. "telnyx".
        digits (str): What to press. Only 0-9, * and # can be pressed, and w waits half a second between two of them.
    """

    vendor: str
    digits: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        vendor = self.vendor

        digits = self.digits

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "vendor": vendor,
                "digits": digits,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        vendor = d.pop("vendor")

        digits = d.pop("digits")

        press_digits_request = cls(
            vendor=vendor,
            digits=digits,
        )

        press_digits_request.additional_properties = d
        return press_digits_request

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
