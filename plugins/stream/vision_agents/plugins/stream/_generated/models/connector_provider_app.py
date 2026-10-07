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

T = TypeVar("T", bound="ConnectorProviderApp")


@_attrs_define
class ConnectorProviderApp:
    """The customer's app at a connector's provider, such as a Slack app. Its client secret and signing secret are kept
    sealed and never returned.

        Attributes:
            client_id (str): The app's OAuth client, which every consent of the connector's connections uses.
            connector_id (str):
            created_at (datetime.datetime):
            provider_app_id (str): The provider's id for the app, such as a Slack app id.
            registration (ConnectorClientRegistrationMethod): operator is this deployment's own client, customer one the app
                registered, managed one the router created for the app (PUT /v1/agents/connectors/{id}/provider-app), dcr one
                registered on the fly (RFC 7591) and cimd one named by a metadata document.
            updated_at (datetime.datetime):
    """

    client_id: str
    connector_id: str
    created_at: datetime.datetime
    provider_app_id: str
    registration: ConnectorClientRegistrationMethod
    updated_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        client_id = self.client_id

        connector_id = self.connector_id

        created_at = self.created_at.isoformat()

        provider_app_id = self.provider_app_id

        registration = self.registration.value

        updated_at = self.updated_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "client_id": client_id,
                "connector_id": connector_id,
                "created_at": created_at,
                "provider_app_id": provider_app_id,
                "registration": registration,
                "updated_at": updated_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        client_id = d.pop("client_id")

        connector_id = d.pop("connector_id")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        provider_app_id = d.pop("provider_app_id")

        registration = ConnectorClientRegistrationMethod(d.pop("registration"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        connector_provider_app = cls(
            client_id=client_id,
            connector_id=connector_id,
            created_at=created_at,
            provider_app_id=provider_app_id,
            registration=registration,
            updated_at=updated_at,
        )

        connector_provider_app.additional_properties = d
        return connector_provider_app

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
