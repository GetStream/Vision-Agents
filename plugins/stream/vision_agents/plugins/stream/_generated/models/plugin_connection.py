from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.plugin_connection_status import PluginConnectionStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.plugin_client import PluginClient


T = TypeVar("T", bound="PluginConnection")


@_attrs_define
class PluginConnection:
    """A catalog plugin as this agent has it, including whether it is logged in. A plugin the config names that nobody has
    logged into yet is not_connected, which is what a dashboard reminds the app to finish, unless it has user, when each
    end user connects it in the conversation.

        Attributes:
            logo_url (str): Where this deployment serves the plugin's logo, as an SVG needing no credential. Empty for an
                MCP server named by URL.
            name (str):
            plugin_id (str):
            status (PluginConnectionStatus): The app's login. Always not_connected for a plugin with user, which the app
                does not log into.
            category (str | Unset):
            client (PluginClient | Unset): The OAuth client a config logs a plugin in with. Its secret is sealed and never
                returned.
            client_required (bool | Unset): True when nobody can connect the plugin until the config has a client of the
                app's own, set with setPluginClient.
            description (str | Unset):
            instance_hint (str | Unset):
            instance_required (bool | Unset):
            instance_url (str | Unset):
            user (bool | Unset): True when the config names the plugin with user: each end user connects their own account
                in the conversation.
    """

    logo_url: str
    name: str
    plugin_id: str
    status: PluginConnectionStatus
    category: str | Unset = UNSET
    client: PluginClient | Unset = UNSET
    client_required: bool | Unset = UNSET
    description: str | Unset = UNSET
    instance_hint: str | Unset = UNSET
    instance_required: bool | Unset = UNSET
    instance_url: str | Unset = UNSET
    user: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        logo_url = self.logo_url

        name = self.name

        plugin_id = self.plugin_id

        status = self.status.value

        category = self.category

        client: dict[str, Any] | Unset = UNSET
        if not isinstance(self.client, Unset):
            client = self.client.to_dict()

        client_required = self.client_required

        description = self.description

        instance_hint = self.instance_hint

        instance_required = self.instance_required

        instance_url = self.instance_url

        user = self.user

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "logo_url": logo_url,
                "name": name,
                "plugin_id": plugin_id,
                "status": status,
            }
        )
        if category is not UNSET:
            field_dict["category"] = category
        if client is not UNSET:
            field_dict["client"] = client
        if client_required is not UNSET:
            field_dict["client_required"] = client_required
        if description is not UNSET:
            field_dict["description"] = description
        if instance_hint is not UNSET:
            field_dict["instance_hint"] = instance_hint
        if instance_required is not UNSET:
            field_dict["instance_required"] = instance_required
        if instance_url is not UNSET:
            field_dict["instance_url"] = instance_url
        if user is not UNSET:
            field_dict["user"] = user

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.plugin_client import PluginClient

        d = dict(src_dict)
        logo_url = d.pop("logo_url")

        name = d.pop("name")

        plugin_id = d.pop("plugin_id")

        status = PluginConnectionStatus(d.pop("status"))

        category = d.pop("category", UNSET)

        _client = d.pop("client", UNSET)
        client: PluginClient | Unset
        if isinstance(_client, Unset):
            client = UNSET
        else:
            client = PluginClient.from_dict(_client)

        client_required = d.pop("client_required", UNSET)

        description = d.pop("description", UNSET)

        instance_hint = d.pop("instance_hint", UNSET)

        instance_required = d.pop("instance_required", UNSET)

        instance_url = d.pop("instance_url", UNSET)

        user = d.pop("user", UNSET)

        plugin_connection = cls(
            logo_url=logo_url,
            name=name,
            plugin_id=plugin_id,
            status=status,
            category=category,
            client=client,
            client_required=client_required,
            description=description,
            instance_hint=instance_hint,
            instance_required=instance_required,
            instance_url=instance_url,
            user=user,
        )

        plugin_connection.additional_properties = d
        return plugin_connection

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
