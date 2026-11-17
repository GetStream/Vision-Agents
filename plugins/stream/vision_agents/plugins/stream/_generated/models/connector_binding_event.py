from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connector_binding_event_arguments import (
        ConnectorBindingEventArguments,
    )


T = TypeVar("T", bound="ConnectorBindingEvent")


@_attrs_define
class ConnectorBindingEvent:
    """One MCP event a binding subscribes to on its fixed connection. Each one that arrives opens a text conversation from
    the config, as the app, with the event's data as the first thing said to it.

        Attributes:
            event (str): The event's name, as the server's events/list gives it, such as issue.created.
            arguments (ConnectorBindingEventArguments | Unset): The event's filters, as its inputSchema describes them.
            instructions (str | Unset): What the agent does with the event when it arrives, added to its instructions for
                that conversation.
    """

    event: str
    arguments: ConnectorBindingEventArguments | Unset = UNSET
    instructions: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        event = self.event

        arguments: dict[str, Any] | Unset = UNSET
        if not isinstance(self.arguments, Unset):
            arguments = self.arguments.to_dict()

        instructions = self.instructions

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "event": event,
            }
        )
        if arguments is not UNSET:
            field_dict["arguments"] = arguments
        if instructions is not UNSET:
            field_dict["instructions"] = instructions

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connector_binding_event_arguments import (
            ConnectorBindingEventArguments,
        )

        d = dict(src_dict)
        event = d.pop("event")

        _arguments = d.pop("arguments", UNSET)
        arguments: ConnectorBindingEventArguments | Unset
        if isinstance(_arguments, Unset):
            arguments = UNSET
        else:
            arguments = ConnectorBindingEventArguments.from_dict(_arguments)

        instructions = d.pop("instructions", UNSET)

        connector_binding_event = cls(
            event=event,
            arguments=arguments,
            instructions=instructions,
        )

        connector_binding_event.additional_properties = d
        return connector_binding_event

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
