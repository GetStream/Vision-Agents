from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.authorization_kind import AuthorizationKind

T = TypeVar("T", bound="Authorization")


@_attrs_define
class Authorization:
    """A consent in flight for one connection: the page that starts it in a browser and the token that binds it to that
    browser.

        Attributes:
            expires_at (datetime.datetime): When the attempt ends, 10 minutes after it began. A callback after that is
                refused.
            handoff_token (str): Handed to the launch page by postMessage, never put in a URL. It binds the attempt to the
                first browser that opens launch_url and hands it off; a second handoff is refused.
            id (str): The attempt.
            kind (AuthorizationKind): consent for a connection no account was connected to yet, reconnect for one that has
                been connected before, which must come back with the same provider account.
            launch_url (str): The router's page to open in a popup from the dashboard. It asks the opener for handoff_token
                and then sends the browser to the provider.
    """

    expires_at: datetime.datetime
    handoff_token: str
    id: str
    kind: AuthorizationKind
    launch_url: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        expires_at = self.expires_at.isoformat()

        handoff_token = self.handoff_token

        id = self.id

        kind = self.kind.value

        launch_url = self.launch_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "expires_at": expires_at,
                "handoff_token": handoff_token,
                "id": id,
                "kind": kind,
                "launch_url": launch_url,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        expires_at = datetime.datetime.fromisoformat(d.pop("expires_at"))

        handoff_token = d.pop("handoff_token")

        id = d.pop("id")

        kind = AuthorizationKind(d.pop("kind"))

        launch_url = d.pop("launch_url")

        authorization = cls(
            expires_at=expires_at,
            handoff_token=handoff_token,
            id=id,
            kind=kind,
            launch_url=launch_url,
        )

        authorization.additional_properties = d
        return authorization

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
