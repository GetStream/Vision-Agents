from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="PutConnectorCredentialsRequest")


@_attrs_define
class PutConnectorCredentialsRequest:
    """
    Attributes:
        expected_revision (int):
        bearer_token (str | Unset): Required for bearer auth; stored encrypted and never returned.
        api_key (str | Unset): Required for api_key auth; stored encrypted and never returned.
        access_token (str | Unset): For OAuth connections, imports an existing provider-issued access token. The token
            endpoint, resource, and client authentication are resolved from the fixed connector definition or discovered
            provider metadata; callers cannot supply an endpoint that receives the token or refresh token.
        refresh_token (str | Unset): Optional OAuth refresh token; stored encrypted with the access token.
        expires_at (datetime.datetime | Unset): Required with an imported OAuth access token.
        granted_scopes (list[str] | Unset): Provider-granted OAuth scopes, validated against the connector catalog where
            known.
        oauth_client_id (str | Unset): Required when importing a grant for a customer-registered or dynamically
            registered OAuth client. Operator-managed providers use their configured client.
        oauth_client_secret (str | Unset): Optional confidential client secret for an imported grant; encrypted at rest.
    """

    expected_revision: int
    bearer_token: str | Unset = UNSET
    api_key: str | Unset = UNSET
    access_token: str | Unset = UNSET
    refresh_token: str | Unset = UNSET
    expires_at: datetime.datetime | Unset = UNSET
    granted_scopes: list[str] | Unset = UNSET
    oauth_client_id: str | Unset = UNSET
    oauth_client_secret: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        expected_revision = self.expected_revision

        bearer_token = self.bearer_token

        api_key = self.api_key

        access_token = self.access_token

        refresh_token = self.refresh_token

        expires_at: str | Unset = UNSET
        if not isinstance(self.expires_at, Unset):
            expires_at = self.expires_at.isoformat()

        granted_scopes: list[str] | Unset = UNSET
        if not isinstance(self.granted_scopes, Unset):
            granted_scopes = self.granted_scopes

        oauth_client_id = self.oauth_client_id

        oauth_client_secret = self.oauth_client_secret

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "expected_revision": expected_revision,
            }
        )
        if bearer_token is not UNSET:
            field_dict["bearer_token"] = bearer_token
        if api_key is not UNSET:
            field_dict["api_key"] = api_key
        if access_token is not UNSET:
            field_dict["access_token"] = access_token
        if refresh_token is not UNSET:
            field_dict["refresh_token"] = refresh_token
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at
        if granted_scopes is not UNSET:
            field_dict["granted_scopes"] = granted_scopes
        if oauth_client_id is not UNSET:
            field_dict["oauth_client_id"] = oauth_client_id
        if oauth_client_secret is not UNSET:
            field_dict["oauth_client_secret"] = oauth_client_secret

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        expected_revision = d.pop("expected_revision")

        bearer_token = d.pop("bearer_token", UNSET)

        api_key = d.pop("api_key", UNSET)

        access_token = d.pop("access_token", UNSET)

        refresh_token = d.pop("refresh_token", UNSET)

        _expires_at = d.pop("expires_at", UNSET)
        expires_at: datetime.datetime | Unset
        if isinstance(_expires_at, Unset):
            expires_at = UNSET
        else:
            expires_at = datetime.datetime.fromisoformat(_expires_at)

        granted_scopes = cast(list[str], d.pop("granted_scopes", UNSET))

        oauth_client_id = d.pop("oauth_client_id", UNSET)

        oauth_client_secret = d.pop("oauth_client_secret", UNSET)

        put_connector_credentials_request = cls(
            expected_revision=expected_revision,
            bearer_token=bearer_token,
            api_key=api_key,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            granted_scopes=granted_scopes,
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
        )

        put_connector_credentials_request.additional_properties = d
        return put_connector_credentials_request

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
