from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorProviderAppFields")


@_attrs_define
class ConnectorProviderAppFields:
    """The customer's own provider app a channel's events are verified with.

    Attributes:
        verify_token (bool): The provider checks the events URL with a verify token before it delivers, as Meta does for
            a webhook: the token is the provider_app_id.
        events_url_prefix (str | Unset): The app's events URL without the app's id at its end: ROUTER_PUBLIC_URL
            followed by /v1/connectors/events/{id}/. Followed by the provider_app_id, it is the URL the provider app
            delivers its events to, such as a Meta app's WhatsApp webhook Callback URL. Absent when ROUTER_PUBLIC_URL is not
            set.
        id_description (str | Unset): Where a person finds the app's id at the provider. Absent when the manifest says
            nothing.
        id_label (str | Unset): How the provider names the app's id, such as Meta App ID. Absent when the manifest says
            nothing.
        secret_description (str | Unset): Where a person finds that secret at the provider. Absent when the manifest
            says nothing.
        secret_label (str | Unset): How the provider names the secret it signs the app's events with, such as Meta App
            Secret. Absent when the manifest says nothing.
    """

    verify_token: bool
    events_url_prefix: str | Unset = UNSET
    id_description: str | Unset = UNSET
    id_label: str | Unset = UNSET
    secret_description: str | Unset = UNSET
    secret_label: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        verify_token = self.verify_token

        events_url_prefix = self.events_url_prefix

        id_description = self.id_description

        id_label = self.id_label

        secret_description = self.secret_description

        secret_label = self.secret_label

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "verify_token": verify_token,
            }
        )
        if events_url_prefix is not UNSET:
            field_dict["events_url_prefix"] = events_url_prefix
        if id_description is not UNSET:
            field_dict["id_description"] = id_description
        if id_label is not UNSET:
            field_dict["id_label"] = id_label
        if secret_description is not UNSET:
            field_dict["secret_description"] = secret_description
        if secret_label is not UNSET:
            field_dict["secret_label"] = secret_label

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        verify_token = d.pop("verify_token")

        events_url_prefix = d.pop("events_url_prefix", UNSET)

        id_description = d.pop("id_description", UNSET)

        id_label = d.pop("id_label", UNSET)

        secret_description = d.pop("secret_description", UNSET)

        secret_label = d.pop("secret_label", UNSET)

        connector_provider_app_fields = cls(
            verify_token=verify_token,
            events_url_prefix=events_url_prefix,
            id_description=id_description,
            id_label=id_label,
            secret_description=secret_description,
            secret_label=secret_label,
        )

        return connector_provider_app_fields
