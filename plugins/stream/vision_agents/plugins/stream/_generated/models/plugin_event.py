from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.plugin_event_arguments import PluginEventArguments


T = TypeVar("T", bound="PluginEvent")


@_attrs_define
class PluginEvent:
    """One MCP event an agent subscribes to on a plugin it names. Each event that arrives opens a text conversation from
    the config, as whoever's login it came through, with the event's data as the first thing said to it.

        Attributes:
            event (str): The event's name, as the server's events/list gives it, such as comment.created.
            plugin (str): A catalog plugin the config names under plugins.
            arguments (PluginEventArguments | Unset): The event's filters, as its inputSchema describes them.
            instructions (str | Unset): What the agent does with the event when it arrives, added to its instructions for
                that conversation.
    """

    event: str
    plugin: str
    arguments: PluginEventArguments | Unset = UNSET
    instructions: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        event = self.event

        plugin = self.plugin

        arguments: dict[str, Any] | Unset = UNSET
        if not isinstance(self.arguments, Unset):
            arguments = self.arguments.to_dict()

        instructions = self.instructions

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "event": event,
                "plugin": plugin,
            }
        )
        if arguments is not UNSET:
            field_dict["arguments"] = arguments
        if instructions is not UNSET:
            field_dict["instructions"] = instructions

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.plugin_event_arguments import PluginEventArguments

        d = dict(src_dict)
        event = d.pop("event")

        plugin = d.pop("plugin")

        _arguments = d.pop("arguments", UNSET)
        arguments: PluginEventArguments | Unset
        if isinstance(_arguments, Unset):
            arguments = UNSET
        else:
            arguments = PluginEventArguments.from_dict(_arguments)

        instructions = d.pop("instructions", UNSET)

        plugin_event = cls(
            event=event,
            plugin=plugin,
            arguments=arguments,
            instructions=instructions,
        )

        plugin_event.additional_properties = d
        return plugin_event

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
