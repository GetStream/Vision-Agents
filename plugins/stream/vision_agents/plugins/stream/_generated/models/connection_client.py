from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connector_client_registration_method import (
    ConnectorClientRegistrationMethod,
)

T = TypeVar("T", bound="ConnectionClient")


@_attrs_define
class ConnectionClient:
    """Which OAuth client a connection's grant was issued to, so a client the router registered on the fly (RFC 7591) can
    be found at the provider. Its secret is never shown.

        Attributes:
            client_id (str): The client identifier, which is not a secret (RFC 6749 section 2.2). For dcr, the one the
                provider issued when the router registered at the consent.
            registration (ConnectorClientRegistrationMethod): operator is this deployment's own client, customer one the app
                registered, managed one the router created for the app (PUT /v1/agents/connectors/{id}/provider-app), dcr one
                registered on the fly (RFC 7591) and cimd one named by a metadata document.
    """

    client_id: str
    registration: ConnectorClientRegistrationMethod
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        client_id = self.client_id

        registration = self.registration.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "client_id": client_id,
                "registration": registration,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        client_id = d.pop("client_id")

        registration = ConnectorClientRegistrationMethod(d.pop("registration"))

        connection_client = cls(
            client_id=client_id,
            registration=registration,
        )

        connection_client.additional_properties = d
        return connection_client

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
