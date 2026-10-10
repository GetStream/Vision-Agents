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

T = TypeVar("T", bound="StoredConnectorOAuthClient")


@_attrs_define
class StoredConnectorOAuthClient:
    """The OAuth client and provider app the router keeps for the app and one connector. It says whether each secret is
    stored, and never carries one.

        Attributes:
            client_id (str): Empty for a provider app without an OAuth client, such as a Linq account.
            connector_id (str):
            created_at (datetime.datetime):
            has_client_secret (bool): A client secret is stored, sealed. False for a public client.
            has_signing_secret (bool): A signing secret for the provider app's events is stored, sealed.
            registration (ConnectorClientRegistrationMethod): operator is this deployment's own client, customer one the app
                registered, managed one the router created for the app (PUT /v1/agents/connectors/{id}/provider-app), dcr one
                registered on the fly (RFC 7591) and cimd one named by a metadata document.
            updated_at (datetime.datetime): When the client, its secrets or its method last changed.
            auth_method (ConnectorOAuthClientAuthMethod | Unset): How the app's own OAuth client authenticates at the token
                endpoint (RFC 7591 section 2): none for a public client, which has no secret, client_secret_basic or
                client_secret_post.
            provider_app_id (str | Unset): The provider's id for the app the client belongs to. Absent when there is none.
    """

    client_id: str
    connector_id: str
    created_at: datetime.datetime
    has_client_secret: bool
    has_signing_secret: bool
    registration: ConnectorClientRegistrationMethod
    updated_at: datetime.datetime
    auth_method: ConnectorOAuthClientAuthMethod | Unset = UNSET
    provider_app_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        client_id = self.client_id

        connector_id = self.connector_id

        created_at = self.created_at.isoformat()

        has_client_secret = self.has_client_secret

        has_signing_secret = self.has_signing_secret

        registration = self.registration.value

        updated_at = self.updated_at.isoformat()

        auth_method: str | Unset = UNSET
        if not isinstance(self.auth_method, Unset):
            auth_method = self.auth_method.value

        provider_app_id = self.provider_app_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "client_id": client_id,
                "connector_id": connector_id,
                "created_at": created_at,
                "has_client_secret": has_client_secret,
                "has_signing_secret": has_signing_secret,
                "registration": registration,
                "updated_at": updated_at,
            }
        )
        if auth_method is not UNSET:
            field_dict["auth_method"] = auth_method
        if provider_app_id is not UNSET:
            field_dict["provider_app_id"] = provider_app_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        client_id = d.pop("client_id")

        connector_id = d.pop("connector_id")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        has_client_secret = d.pop("has_client_secret")

        has_signing_secret = d.pop("has_signing_secret")

        registration = ConnectorClientRegistrationMethod(d.pop("registration"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        _auth_method = d.pop("auth_method", UNSET)
        auth_method: ConnectorOAuthClientAuthMethod | Unset
        if isinstance(_auth_method, Unset):
            auth_method = UNSET
        else:
            auth_method = ConnectorOAuthClientAuthMethod(_auth_method)

        provider_app_id = d.pop("provider_app_id", UNSET)

        stored_connector_o_auth_client = cls(
            client_id=client_id,
            connector_id=connector_id,
            created_at=created_at,
            has_client_secret=has_client_secret,
            has_signing_secret=has_signing_secret,
            registration=registration,
            updated_at=updated_at,
            auth_method=auth_method,
            provider_app_id=provider_app_id,
        )

        stored_connector_o_auth_client.additional_properties = d
        return stored_connector_o_auth_client

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
