from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="CreateSipTrunkRequest")


@_attrs_define
class CreateSipTrunkRequest:
    """
    Attributes:
        codecs (list[str] | Unset): PCMU, PCMA or G722, in order of preference. Omit for PCMU then PCMA.
        host (str | Unset): Required. The trunk's hostname, without sip: or a port, e.g. example.pstn.twilio.com.
        late_offer (bool | Unset): The trunk accepts an INVITE without SDP. Omit for false.
        name (str | Unset): Required.
        password (str | Unset): Required. Stored sealed and never returned.
        port (int | Unset): Omit for 5060.
        transport (str | Unset): udp, tcp or tls. Omit for tcp.
        username (str | Unset): Required.
    """

    codecs: list[str] | Unset = UNSET
    host: str | Unset = UNSET
    late_offer: bool | Unset = UNSET
    name: str | Unset = UNSET
    password: str | Unset = UNSET
    port: int | Unset = UNSET
    transport: str | Unset = UNSET
    username: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        codecs: list[str] | Unset = UNSET
        if not isinstance(self.codecs, Unset):
            codecs = self.codecs

        host = self.host

        late_offer = self.late_offer

        name = self.name

        password = self.password

        port = self.port

        transport = self.transport

        username = self.username

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if codecs is not UNSET:
            field_dict["codecs"] = codecs
        if host is not UNSET:
            field_dict["host"] = host
        if late_offer is not UNSET:
            field_dict["late_offer"] = late_offer
        if name is not UNSET:
            field_dict["name"] = name
        if password is not UNSET:
            field_dict["password"] = password
        if port is not UNSET:
            field_dict["port"] = port
        if transport is not UNSET:
            field_dict["transport"] = transport
        if username is not UNSET:
            field_dict["username"] = username

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        codecs = cast(list[str], d.pop("codecs", UNSET))

        host = d.pop("host", UNSET)

        late_offer = d.pop("late_offer", UNSET)

        name = d.pop("name", UNSET)

        password = d.pop("password", UNSET)

        port = d.pop("port", UNSET)

        transport = d.pop("transport", UNSET)

        username = d.pop("username", UNSET)

        create_sip_trunk_request = cls(
            codecs=codecs,
            host=host,
            late_offer=late_offer,
            name=name,
            password=password,
            port=port,
            transport=transport,
            username=username,
        )

        create_sip_trunk_request.additional_properties = d
        return create_sip_trunk_request

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
