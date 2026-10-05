from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="Plugin")


@_attrs_define
class Plugin:
    """One hosted MCP server from the built-in catalog.

    Attributes:
        category (str):
        description (str):
        id (str):
        logo_url (str): Where this deployment serves the plugin's logo, as an SVG needing no credential.
        name (str):
        instance_hint (str | Unset):
        instance_required (bool | Unset):
        readonly (bool | Unset): True when the plugin has a read-only endpoint an agent may pick in plugin_options.
        scopes_supported (list[str] | Unset): The OAuth scopes an agent may ask for in plugin_options, as the server
            advertises them. Absent when the server says nothing, and any scope is then passed through.
        toolsets (list[str] | Unset): The groups of tools an agent may limit the plugin to in plugin_options. Absent
            when it cannot be limited.
    """

    category: str
    description: str
    id: str
    logo_url: str
    name: str
    instance_hint: str | Unset = UNSET
    instance_required: bool | Unset = UNSET
    readonly: bool | Unset = UNSET
    scopes_supported: list[str] | Unset = UNSET
    toolsets: list[str] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        category = self.category

        description = self.description

        id = self.id

        logo_url = self.logo_url

        name = self.name

        instance_hint = self.instance_hint

        instance_required = self.instance_required

        readonly = self.readonly

        scopes_supported: list[str] | Unset = UNSET
        if not isinstance(self.scopes_supported, Unset):
            scopes_supported = self.scopes_supported

        toolsets: list[str] | Unset = UNSET
        if not isinstance(self.toolsets, Unset):
            toolsets = self.toolsets

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "category": category,
                "description": description,
                "id": id,
                "logo_url": logo_url,
                "name": name,
            }
        )
        if instance_hint is not UNSET:
            field_dict["instance_hint"] = instance_hint
        if instance_required is not UNSET:
            field_dict["instance_required"] = instance_required
        if readonly is not UNSET:
            field_dict["readonly"] = readonly
        if scopes_supported is not UNSET:
            field_dict["scopes_supported"] = scopes_supported
        if toolsets is not UNSET:
            field_dict["toolsets"] = toolsets

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        category = d.pop("category")

        description = d.pop("description")

        id = d.pop("id")

        logo_url = d.pop("logo_url")

        name = d.pop("name")

        instance_hint = d.pop("instance_hint", UNSET)

        instance_required = d.pop("instance_required", UNSET)

        readonly = d.pop("readonly", UNSET)

        scopes_supported = cast(list[str], d.pop("scopes_supported", UNSET))

        toolsets = cast(list[str], d.pop("toolsets", UNSET))

        plugin = cls(
            category=category,
            description=description,
            id=id,
            logo_url=logo_url,
            name=name,
            instance_hint=instance_hint,
            instance_required=instance_required,
            readonly=readonly,
            scopes_supported=scopes_supported,
            toolsets=toolsets,
        )

        plugin.additional_properties = d
        return plugin

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
