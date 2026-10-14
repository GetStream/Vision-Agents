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
        status (str): The vendor's own word for where the call is, e.g. "queued".
        vendor_call_id (str):
        session_id (str | Unset): The session to open, with start_voice, for the call: the answered leg is routed into
            agent:<session id>, and an agent that is not in it hears nothing when the person picks up.
        vendor (str | Unset): Who is placing the call.
    """

    status: str
    vendor_call_id: str
    session_id: str | Unset = UNSET
    vendor: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        vendor_call_id = self.vendor_call_id

        session_id = self.session_id

        vendor = self.vendor

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
                "vendor_call_id": vendor_call_id,
            }
        )
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
        if vendor is not UNSET:
            field_dict["vendor"] = vendor

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        status = d.pop("status")

        vendor_call_id = d.pop("vendor_call_id")

        session_id = d.pop("session_id", UNSET)

        vendor = d.pop("vendor", UNSET)

        placed_call = cls(
            status=status,
            vendor_call_id=vendor_call_id,
            session_id=session_id,
            vendor=vendor,
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
