from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="AttachedNumber")


@_attrs_define
class AttachedNumber:
    """
    Attributes:
        trunk_id (str):
        route_id (str):
        sip_uri (str): Where the vendor sends calls, e.g. sip:trunk@sip.stream-io-api.com.
    """

    trunk_id: str
    route_id: str
    sip_uri: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        trunk_id = self.trunk_id

        route_id = self.route_id

        sip_uri = self.sip_uri

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "trunk_id": trunk_id,
                "route_id": route_id,
                "sip_uri": sip_uri,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        trunk_id = d.pop("trunk_id")

        route_id = d.pop("route_id")

        sip_uri = d.pop("sip_uri")

        attached_number = cls(
            trunk_id=trunk_id,
            route_id=route_id,
            sip_uri=sip_uri,
        )

        attached_number.additional_properties = d
        return attached_number

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
