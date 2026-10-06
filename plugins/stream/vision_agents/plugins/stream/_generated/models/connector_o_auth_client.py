from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connector_client_registration_method import (
    ConnectorClientRegistrationMethod,
)
from ..models.connector_o_auth_client_auth_method import ConnectorOAuthClientAuthMethod
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorOAuthClient")


@_attrs_define
class ConnectorOAuthClient:
    """The OAuth client the app registered with a connector's provider itself. The secret is write-only: no response
    carries it.

        Attributes:
            client_id (str):
            connector_id (str):
            created_at (datetime.datetime):
            registration (ConnectorClientRegistrationMethod): operator is this deployment's own client, customer one the app
                registered, dcr one registered on the fly (RFC 7591) and cimd one named by a metadata document.
            updated_at (datetime.datetime): When the client, its secret or its method last changed.
            auth_method (ConnectorOAuthClientAuthMethod | Unset): How the app's own OAuth client authenticates at the token
                endpoint (RFC 7591 section 2): none for a public client, which has no secret, client_secret_basic or
                client_secret_post.
    """

    client_id: str
    connector_id: str
    created_at: datetime.datetime
    registration: ConnectorClientRegistrationMethod
    updated_at: datetime.datetime
    auth_method: ConnectorOAuthClientAuthMethod | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        client_id = self.client_id

        connector_id = self.connector_id

        created_at = self.created_at.isoformat()

        registration = self.registration.value

        updated_at = self.updated_at.isoformat()

        auth_method: str | Unset = UNSET
        if not isinstance(self.auth_method, Unset):
            auth_method = self.auth_method.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "client_id": client_id,
                "connector_id": connector_id,
                "created_at": created_at,
                "registration": registration,
                "updated_at": updated_at,
            }
        )
        if auth_method is not UNSET:
            field_dict["auth_method"] = auth_method

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        client_id = d.pop("client_id")

        connector_id = d.pop("connector_id")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        registration = ConnectorClientRegistrationMethod(d.pop("registration"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        _auth_method = d.pop("auth_method", UNSET)
        auth_method: ConnectorOAuthClientAuthMethod | Unset
        if isinstance(_auth_method, Unset):
            auth_method = UNSET
        else:
            auth_method = ConnectorOAuthClientAuthMethod(_auth_method)

        connector_o_auth_client = cls(
            client_id=client_id,
            connector_id=connector_id,
            created_at=created_at,
            registration=registration,
            updated_at=updated_at,
            auth_method=auth_method,
        )

        connector_o_auth_client.additional_properties = d
        return connector_o_auth_client

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
