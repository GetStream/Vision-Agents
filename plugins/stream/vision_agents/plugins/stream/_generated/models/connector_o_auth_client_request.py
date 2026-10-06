from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.connector_o_auth_client_auth_method import ConnectorOAuthClientAuthMethod
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorOAuthClientRequest")


@_attrs_define
class ConnectorOAuthClientRequest:
    """The OAuth client the app registered with the connector's provider. An unknown field is refused rather than ignored.

    Attributes:
        client_id (str):
        auth_method (ConnectorOAuthClientAuthMethod | Unset): How the app's own OAuth client authenticates at the token
            endpoint (RFC 7591 section 2): none for a public client, which has no secret, client_secret_basic or
            client_secret_post.
        client_secret (str | Unset): Sealed at rest and never returned. Left out for a public client (auth_method none).
    """

    client_id: str
    auth_method: ConnectorOAuthClientAuthMethod | Unset = UNSET
    client_secret: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        client_id = self.client_id

        auth_method: str | Unset = UNSET
        if not isinstance(self.auth_method, Unset):
            auth_method = self.auth_method.value

        client_secret = self.client_secret

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "client_id": client_id,
            }
        )
        if auth_method is not UNSET:
            field_dict["auth_method"] = auth_method
        if client_secret is not UNSET:
            field_dict["client_secret"] = client_secret

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        client_id = d.pop("client_id")

        _auth_method = d.pop("auth_method", UNSET)
        auth_method: ConnectorOAuthClientAuthMethod | Unset
        if isinstance(_auth_method, Unset):
            auth_method = UNSET
        else:
            auth_method = ConnectorOAuthClientAuthMethod(_auth_method)

        client_secret = d.pop("client_secret", UNSET)

        connector_o_auth_client_request = cls(
            client_id=client_id,
            auth_method=auth_method,
            client_secret=client_secret,
        )

        return connector_o_auth_client_request
