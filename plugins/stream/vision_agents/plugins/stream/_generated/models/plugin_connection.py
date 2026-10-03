from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.plugin_connection_status import PluginConnectionStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="PluginConnection")


@_attrs_define
class PluginConnection:
    """A catalog plugin as this agent has it, including whether it is logged in. A plugin the config names that nobody has
    logged into yet is not_connected, which is what a dashboard reminds the app to finish.

        Attributes:
            name (str):
            plugin_id (str):
            status (PluginConnectionStatus):
            category (str | Unset):
            description (str | Unset):
            instance_hint (str | Unset):
            instance_required (bool | Unset):
            instance_url (str | Unset):
    """

    name: str
    plugin_id: str
    status: PluginConnectionStatus
    category: str | Unset = UNSET
    description: str | Unset = UNSET
    instance_hint: str | Unset = UNSET
    instance_required: bool | Unset = UNSET
    instance_url: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        plugin_id = self.plugin_id

        status = self.status.value

        category = self.category

        description = self.description

        instance_hint = self.instance_hint

        instance_required = self.instance_required

        instance_url = self.instance_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "plugin_id": plugin_id,
                "status": status,
            }
        )
        if category is not UNSET:
            field_dict["category"] = category
        if description is not UNSET:
            field_dict["description"] = description
        if instance_hint is not UNSET:
            field_dict["instance_hint"] = instance_hint
        if instance_required is not UNSET:
            field_dict["instance_required"] = instance_required
        if instance_url is not UNSET:
            field_dict["instance_url"] = instance_url

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        plugin_id = d.pop("plugin_id")

        status = PluginConnectionStatus(d.pop("status"))

        category = d.pop("category", UNSET)

        description = d.pop("description", UNSET)

        instance_hint = d.pop("instance_hint", UNSET)

        instance_required = d.pop("instance_required", UNSET)

        instance_url = d.pop("instance_url", UNSET)

        plugin_connection = cls(
            name=name,
            plugin_id=plugin_id,
            status=status,
            category=category,
            description=description,
            instance_hint=instance_hint,
            instance_required=instance_required,
            instance_url=instance_url,
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
