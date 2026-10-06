from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SetPluginClientRequest")


@_attrs_define
class SetPluginClientRequest:
    """The OAuth client an app registered with a plugin's provider, with the redirect URI <public
    url>/v1/agents/plugins/callback.

        Attributes:
            client_id (str): The client id the provider issued.
            client_secret (str | Unset): The client secret the provider issued. Left out for a public client.
            user (bool | Unset): Also name the plugin under the config's user_plugins, so that each end user connects their
                own account in the conversation, the first time the agent needs it. Left out names nothing: the app connects the
                plugin once with authorize, which names it under agent_plugins.
    """

    client_id: str
    client_secret: str | Unset = UNSET
    user: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        client_id = self.client_id

        client_secret = self.client_secret

        user = self.user

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "client_id": client_id,
            }
        )
        if client_secret is not UNSET:
            field_dict["client_secret"] = client_secret
        if user is not UNSET:
            field_dict["user"] = user

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        client_id = d.pop("client_id")

        client_secret = d.pop("client_secret", UNSET)

        user = d.pop("user", UNSET)

        set_plugin_client_request = cls(
            client_id=client_id,
            client_secret=client_secret,
            user=user,
        )

        set_plugin_client_request.additional_properties = d
        return set_plugin_client_request

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
