from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.connector_client_alg import ConnectorClientAlg
from ..models.connector_client_auth_method import ConnectorClientAuthMethod
from ..models.connector_client_registration import ConnectorClientRegistration
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorClient")


@_attrs_define
class ConnectorClient:
    """How the OAuth client a connection uses is registered, and how the client authenticates at the token endpoint.

    Attributes:
        alg (ConnectorClientAlg | Unset): How a private_key_jwt assertion is signed, and set only for it.
        auth_method (ConnectorClientAuthMethod | Unset): How the OAuth client authenticates at the token endpoint, as
            the IANA OAuth token endpoint authentication methods registry spells it.
        registration (list[ConnectorClientRegistration] | None | Unset): The client registration mechanisms the
            connector allows, tried as the scheme orders them. Empty when the connector needs no OAuth client.
    """

    alg: ConnectorClientAlg | Unset = UNSET
    auth_method: ConnectorClientAuthMethod | Unset = UNSET
    registration: list[ConnectorClientRegistration] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        alg: str | Unset = UNSET
        if not isinstance(self.alg, Unset):
            alg = self.alg.value

        auth_method: str | Unset = UNSET
        if not isinstance(self.auth_method, Unset):
            auth_method = self.auth_method.value

        registration: list[str] | None | Unset
        if isinstance(self.registration, Unset):
            registration = UNSET
        elif isinstance(self.registration, list):
            registration = []
            for registration_type_0_item_data in self.registration:
                registration_type_0_item = registration_type_0_item_data.value
                registration.append(registration_type_0_item)

        else:
            registration = self.registration

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if alg is not UNSET:
            field_dict["alg"] = alg
        if auth_method is not UNSET:
            field_dict["auth_method"] = auth_method
        if registration is not UNSET:
            field_dict["registration"] = registration

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        _alg = d.pop("alg", UNSET)
        alg: ConnectorClientAlg | Unset
        if isinstance(_alg, Unset):
            alg = UNSET
        else:
            alg = ConnectorClientAlg(_alg)

        _auth_method = d.pop("auth_method", UNSET)
        auth_method: ConnectorClientAuthMethod | Unset
        if isinstance(_auth_method, Unset):
            auth_method = UNSET
        else:
            auth_method = ConnectorClientAuthMethod(_auth_method)

        def _parse_registration(
            data: object,
        ) -> list[ConnectorClientRegistration] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                registration_type_0 = []
                _registration_type_0 = data
                for registration_type_0_item_data in _registration_type_0:
                    registration_type_0_item = ConnectorClientRegistration(
                        registration_type_0_item_data
                    )

                    registration_type_0.append(registration_type_0_item)

                return registration_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ConnectorClientRegistration] | None | Unset, data)

        registration = _parse_registration(d.pop("registration", UNSET))

        connector_client = cls(
            alg=alg,
            auth_method=auth_method,
            registration=registration,
        )

        return connector_client
