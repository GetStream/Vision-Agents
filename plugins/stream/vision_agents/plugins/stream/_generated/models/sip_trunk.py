from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.sip_trunk_transport import SipTrunkTransport

T = TypeVar("T", bound="SipTrunk")


@_attrs_define
class SipTrunk:
    """
    Attributes:
        codecs (list[str]): Audio codecs offered to the trunk, in order of preference.
        created_at (datetime.datetime):
        has_password (bool): Whether a password is stored. The password itself is never returned. False for a trunk that
            arrived from another deployment, which needs one set before it can be called through.
        host (str): The trunk's hostname, without sip: or a port.
        id (str):
        late_offer (bool): The trunk accepts an INVITE without SDP.
        name (str):
        port (int):
        transport (SipTrunkTransport):
        updated_at (datetime.datetime):
        username (str):
    """

    codecs: list[str]
    created_at: datetime.datetime
    has_password: bool
    host: str
    id: str
    late_offer: bool
    name: str
    port: int
    transport: SipTrunkTransport
    updated_at: datetime.datetime
    username: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        codecs = self.codecs

        created_at = self.created_at.isoformat()

        has_password = self.has_password

        host = self.host

        id = self.id

        late_offer = self.late_offer

        name = self.name

        port = self.port

        transport = self.transport.value

        updated_at = self.updated_at.isoformat()

        username = self.username

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "codecs": codecs,
                "created_at": created_at,
                "has_password": has_password,
                "host": host,
                "id": id,
                "late_offer": late_offer,
                "name": name,
                "port": port,
                "transport": transport,
                "updated_at": updated_at,
                "username": username,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        codecs = cast(list[str], d.pop("codecs"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        has_password = d.pop("has_password")

        host = d.pop("host")

        id = d.pop("id")

        late_offer = d.pop("late_offer")

        name = d.pop("name")

        port = d.pop("port")

        transport = SipTrunkTransport(d.pop("transport"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        username = d.pop("username")

        sip_trunk = cls(
            codecs=codecs,
            created_at=created_at,
            has_password=has_password,
            host=host,
            id=id,
            late_offer=late_offer,
            name=name,
            port=port,
            transport=transport,
            updated_at=updated_at,
            username=username,
        )

        sip_trunk.additional_properties = d
        return sip_trunk

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
