from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectionToken")


@_attrs_define
class ConnectionToken:
    """A connection's access credential, for the app's backend to call the provider with directly. It holds no refresh
    token.

        Attributes:
            connection_id (str):
            header (str): The HTTP field to send it in: Authorization for an OAuth access token, the connection's own header
                for an API key.
            value (str): The whole field value: Bearer and the access token for an OAuth access token (RFC 6750 section
                2.1), the key for an API key.
            expires_at (datetime.datetime | Unset): When it stops working. Absent when the provider gave no expiry. Export
                again for a fresh one.
    """

    connection_id: str
    header: str
    value: str
    expires_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connection_id = self.connection_id

        header = self.header

        value = self.value

        expires_at: str | Unset = UNSET
        if not isinstance(self.expires_at, Unset):
            expires_at = self.expires_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection_id": connection_id,
                "header": header,
                "value": value,
            }
        )
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        header = d.pop("header")

        value = d.pop("value")

        _expires_at = d.pop("expires_at", UNSET)
        expires_at: datetime.datetime | Unset
        if isinstance(_expires_at, Unset):
            expires_at = UNSET
        else:
            expires_at = datetime.datetime.fromisoformat(_expires_at)

        connection_token = cls(
            connection_id=connection_id,
            header=header,
            value=value,
            expires_at=expires_at,
        )

        connection_token.additional_properties = d
        return connection_token

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
