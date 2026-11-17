from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ConnectionCredentialsValues")


@_attrs_define
class ConnectionCredentialsValues:
    """What the connection's auth_scheme takes, write-only. api_key: api_key and header. bearer: token. none: nothing,
    which activates the connection. oauth2_client_credentials: client_id and client_secret, which are tried at the token
    endpoint at once. oauth2_code: a grant the provider already issued, as access_token, refresh_token (optional),
    expires_at (RFC 3339) and scope (the granted scopes joined as the connector's scopes are); its endpoints and client
    are the connector's, never the caller's.

    """

    additional_properties: dict[str, str] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        connection_credentials_values = cls()

        connection_credentials_values.additional_properties = d
        return connection_credentials_values

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> str:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
