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
        auth_method (ConnectorOAuthClientAuthMethod | Unset): How the app's own OAuth client authenticates at the token
            endpoint (RFC 7591 section 2): none for a public client, which has no secret, client_secret_basic or
            client_secret_post.
        client_id (str | Unset): Required, unless the record is only a provider app: provider_app_id and signing_secret
            without client_secret or auth_method, for a connector whose connections are not consented through oauth2_code,
            such as linq.
        client_secret (str | Unset): Sealed at rest and never returned. Left out for a public client (auth_method none).
        provider_app_id (str | Unset): The provider's id for the app the client belongs to, such as a Slack app id
            (A012ABCD0A0). The app's events then reach POST /v1/connectors/events/{id}/{provider_app_id}. An app serves one
            customer: another customer's record naming it is a 409.
        signing_secret (str | Unset): The secret the provider signs the app's events with, such as a Slack app's signing
            secret. Needs provider_app_id, and a connector whose events are verified with the app's own secret
            (channel.verifier.secret provider_app). Sealed at rest and never returned. Putting the client again without it
            removes it, as it does client_secret.
    """

    auth_method: ConnectorOAuthClientAuthMethod | Unset = UNSET
    client_id: str | Unset = UNSET
    client_secret: str | Unset = UNSET
    provider_app_id: str | Unset = UNSET
    signing_secret: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        auth_method: str | Unset = UNSET
        if not isinstance(self.auth_method, Unset):
            auth_method = self.auth_method.value

        client_id = self.client_id

        client_secret = self.client_secret

        provider_app_id = self.provider_app_id

        signing_secret = self.signing_secret

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if auth_method is not UNSET:
            field_dict["auth_method"] = auth_method
        if client_id is not UNSET:
            field_dict["client_id"] = client_id
        if client_secret is not UNSET:
            field_dict["client_secret"] = client_secret
        if provider_app_id is not UNSET:
            field_dict["provider_app_id"] = provider_app_id
        if signing_secret is not UNSET:
            field_dict["signing_secret"] = signing_secret

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        _auth_method = d.pop("auth_method", UNSET)
        auth_method: ConnectorOAuthClientAuthMethod | Unset
        if isinstance(_auth_method, Unset):
            auth_method = UNSET
        else:
            auth_method = ConnectorOAuthClientAuthMethod(_auth_method)

        client_id = d.pop("client_id", UNSET)

        client_secret = d.pop("client_secret", UNSET)

        provider_app_id = d.pop("provider_app_id", UNSET)

        signing_secret = d.pop("signing_secret", UNSET)

        connector_o_auth_client_request = cls(
            auth_method=auth_method,
            client_id=client_id,
            client_secret=client_secret,
            provider_app_id=provider_app_id,
            signing_secret=signing_secret,
        )

        return connector_o_auth_client_request
