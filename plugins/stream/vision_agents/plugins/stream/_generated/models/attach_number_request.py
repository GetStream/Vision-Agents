from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AttachNumberRequest")


@_attrs_define
class AttachNumberRequest:
    """
    Attributes:
        call_id (str | Unset): The call every caller joins. Omit to give each caller their own call, named after the
            number they rang.
        call_type (str | Unset): The Stream call type. Omit for "default".
        allowed_ips (list[str] | Unset): The vendor's signalling addresses, as IPs or CIDR blocks.
    """

    call_id: str | Unset = UNSET
    call_type: str | Unset = UNSET
    allowed_ips: list[str] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        call_id = self.call_id

        call_type = self.call_type

        allowed_ips: list[str] | Unset = UNSET
        if not isinstance(self.allowed_ips, Unset):
            allowed_ips = self.allowed_ips

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if call_type is not UNSET:
            field_dict["call_type"] = call_type
        if allowed_ips is not UNSET:
            field_dict["allowed_ips"] = allowed_ips

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        call_id = d.pop("call_id", UNSET)

        call_type = d.pop("call_type", UNSET)

        allowed_ips = cast(list[str], d.pop("allowed_ips", UNSET))

        attach_number_request = cls(
            call_id=call_id,
            call_type=call_type,
            allowed_ips=allowed_ips,
        )

        attach_number_request.additional_properties = d
        return attach_number_request

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
