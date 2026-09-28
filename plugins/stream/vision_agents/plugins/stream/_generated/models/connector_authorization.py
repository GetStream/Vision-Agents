from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ConnectorAuthorization")


@_attrs_define
class ConnectorAuthorization:
    """
    Attributes:
        authorization_url (str):
        authorization_id (str):
        handoff_token (str): Short-lived secret delivered only to the router launch popup through postMessage. Never
            place it in a URL or persist it in application storage.
        expires_at (datetime.datetime):
    """

    authorization_url: str
    authorization_id: str
    handoff_token: str
    expires_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        authorization_url = self.authorization_url

        authorization_id = self.authorization_id

        handoff_token = self.handoff_token

        expires_at = self.expires_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "authorization_url": authorization_url,
                "authorization_id": authorization_id,
                "handoff_token": handoff_token,
                "expires_at": expires_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        authorization_url = d.pop("authorization_url")

        authorization_id = d.pop("authorization_id")

        handoff_token = d.pop("handoff_token")

        expires_at = datetime.datetime.fromisoformat(d.pop("expires_at"))

        connector_authorization = cls(
            authorization_url=authorization_url,
            authorization_id=authorization_id,
            handoff_token=handoff_token,
            expires_at=expires_at,
        )

        connector_authorization.additional_properties = d
        return connector_authorization

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
