from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="PluginWithOptions")


@_attrs_define
class PluginWithOptions:
    """One catalog plugin an agent names, with how it is reached and what its login asks for. A login made before a change
    keeps what it was granted, so connect it again for the change to take.

        Attributes:
            name (str): A catalog plugin id, such as linear.
            readonly (bool | Unset): Reach the plugin's read-only MCP endpoint, which offers no tool that writes and asks
                for read access at consent. Only a plugin whose vendor runs one may set it, such as linear.
            scopes (list[str] | Unset): The OAuth scopes asked for at consent, in place of the catalog's. Left out asks for
                the catalog's, or the read-only endpoint's when readonly is set.
            tools (list[str] | Unset): Offer the model only the server's tools matching these names or path.Match patterns,
                such as search_files or read_*. A tool left out is neither listed nor callable. Left out offers every tool.
            toolsets (list[str] | Unset): Limit the server to these groups of tools, from the plugin's toolsets in the
                catalog, such as calcom's bookings and availability. Left out offers every tool. Changing them needs no new
                login.
            user (bool | Unset): Each end user connects the plugin with their own account, in the conversation, the first
                time the agent needs it, as a plugin_authorization attachment. Left out, the catalog decides: a plugin reaching
                a person's own account, such as google_calendar, is connected by each end user, and one reaching the company's,
                such as sentry, by the app once, from the dashboard. false has the app connect it whatever the catalog says.
    """

    name: str
    readonly: bool | Unset = UNSET
    scopes: list[str] | Unset = UNSET
    tools: list[str] | Unset = UNSET
    toolsets: list[str] | Unset = UNSET
    user: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

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

        user = self.user

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
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
        if user is not UNSET:
            field_dict["user"] = user

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        readonly = d.pop("readonly", UNSET)

        scopes = cast(list[str], d.pop("scopes", UNSET))

        tools = cast(list[str], d.pop("tools", UNSET))

        toolsets = cast(list[str], d.pop("toolsets", UNSET))

        user = d.pop("user", UNSET)

        plugin_with_options = cls(
            name=name,
            readonly=readonly,
            scopes=scopes,
            tools=tools,
            toolsets=toolsets,
            user=user,
        )

        plugin_with_options.additional_properties = d
        return plugin_with_options

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
