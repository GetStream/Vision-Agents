from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connection_credentials_values import ConnectionCredentialsValues


T = TypeVar("T", bound="ConnectionCredentials")


@_attrs_define
class ConnectionCredentials:
    """Credentials for a connection, under the revision the caller last read. An unknown field is refused rather than
    ignored.

        Attributes:
            expected_revision (int): The connection's revision as last read. A connection that has moved past it is refused
                with a 409, so two writers never replace each other's credentials unseen.
            values (ConnectionCredentialsValues | Unset): What the connection's auth_scheme takes, write-only. api_key:
                api_key and header. bearer: token. none: nothing, which activates the connection. oauth2_client_credentials:
                client_id and client_secret, which are tried at the token endpoint at once. oauth2_code: a grant the provider
                already issued, as access_token, refresh_token (optional), expires_at (RFC 3339) and scope (the granted scopes
                joined as the connector's scopes are); its endpoints and client are the connector's, never the caller's.
    """

    expected_revision: int
    values: ConnectionCredentialsValues | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        expected_revision = self.expected_revision

        values: dict[str, Any] | Unset = UNSET
        if not isinstance(self.values, Unset):
            values = self.values.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "expected_revision": expected_revision,
            }
        )
        if values is not UNSET:
            field_dict["values"] = values

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connection_credentials_values import ConnectionCredentialsValues

        d = dict(src_dict)
        expected_revision = d.pop("expected_revision")

        _values = d.pop("values", UNSET)
        values: ConnectionCredentialsValues | Unset
        if isinstance(_values, Unset):
            values = UNSET
        else:
            values = ConnectionCredentialsValues.from_dict(_values)

        connection_credentials = cls(
            expected_revision=expected_revision,
            values=values,
        )

        return connection_credentials
