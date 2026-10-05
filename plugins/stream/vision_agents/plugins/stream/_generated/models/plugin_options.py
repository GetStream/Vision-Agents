from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="PluginOptions")


@_attrs_define
class PluginOptions:
    """How an agent reaches one catalog plugin it names, and what its login asks for. A login made before a change keeps
    what it was granted, so connect it again for the change to take.

        Attributes:
            plugin (str): A catalog plugin id. It applies once the config names the plugin under plugins or user_plugins,
                and to the app's login made from the dashboard.
            readonly (bool | Unset): Reach the plugin's read-only MCP endpoint, which offers no tool that writes and asks
                for read access at consent. Only a plugin whose vendor runs one may set it, such as linear.
            scopes (list[str] | Unset): The OAuth scopes asked for at consent, in place of the catalog's. Left out asks for
                the catalog's, or the read-only endpoint's when readonly is set.
            tools (list[str] | Unset): Offer the model only the server's tools matching these names or path.Match patterns,
                such as search_files or read_*. A tool left out is neither listed nor callable. Left out offers every tool.
            toolsets (list[str] | Unset): Limit the server to these groups of tools, from the plugin's toolsets in the
                catalog, such as calcom's bookings and availability. Left out offers every tool. Changing them needs no new
                login.
    """

    plugin: str
    readonly: bool | Unset = UNSET
    scopes: list[str] | Unset = UNSET
    tools: list[str] | Unset = UNSET
    toolsets: list[str] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        plugin = self.plugin

        readonly = self.readonly

        scopes: list[str] | Unset = UNSET
        if not isinstance(self.scopes, Unset):
            scopes = self.scopes

        tools: list[str] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = self.tools

        toolsets: list[str] | Unset = UNSET
        if not isinstance(self.toolsets, Unset):
            toolsets = self.toolsets

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "plugin": plugin,
            }
        )
        if readonly is not UNSET:
            field_dict["readonly"] = readonly
        if scopes is not UNSET:
            field_dict["scopes"] = scopes
        if tools is not UNSET:
            field_dict["tools"] = tools
        if toolsets is not UNSET:
            field_dict["toolsets"] = toolsets

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        plugin = d.pop("plugin")

        readonly = d.pop("readonly", UNSET)

        scopes = cast(list[str], d.pop("scopes", UNSET))

        tools = cast(list[str], d.pop("tools", UNSET))

        toolsets = cast(list[str], d.pop("toolsets", UNSET))

        plugin_options = cls(
            plugin=plugin,
            readonly=readonly,
            scopes=scopes,
            tools=tools,
            toolsets=toolsets,
        )

        plugin_options.additional_properties = d
        return plugin_options

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
