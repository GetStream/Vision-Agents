from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AuthorizeConnectorRequest")


@_attrs_define
class AuthorizeConnectorRequest:
    """
    Attributes:
        scopes (list[str] | Unset): Minimum provider permissions to request. For Slack, choose only scopes needed by
            this connection; values must be in the connector definition.
        oauth_client_id (str | Unset): Supply with oauth_client_secret for a manually registered integration. Some
            providers also support dynamic registration when both are omitted.
        oauth_client_secret (str | Unset): Supply with oauth_client_id for a manually registered integration. Write-
            only; encrypted at rest and used only for OAuth exchange and refresh.
    """

    scopes: list[str] | Unset = UNSET
    oauth_client_id: str | Unset = UNSET
    oauth_client_secret: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        scopes: list[str] | Unset = UNSET
        if not isinstance(self.scopes, Unset):
            scopes = self.scopes

        oauth_client_id = self.oauth_client_id

        oauth_client_secret = self.oauth_client_secret

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if scopes is not UNSET:
            field_dict["scopes"] = scopes
        if oauth_client_id is not UNSET:
            field_dict["oauth_client_id"] = oauth_client_id
        if oauth_client_secret is not UNSET:
            field_dict["oauth_client_secret"] = oauth_client_secret

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        scopes = cast(list[str], d.pop("scopes", UNSET))

        oauth_client_id = d.pop("oauth_client_id", UNSET)

        oauth_client_secret = d.pop("oauth_client_secret", UNSET)

        authorize_connector_request = cls(
            scopes=scopes,
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
        )

        authorize_connector_request.additional_properties = d
        return authorize_connector_request

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
