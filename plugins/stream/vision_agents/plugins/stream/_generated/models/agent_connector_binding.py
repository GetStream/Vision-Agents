from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_connector_selection import AgentConnectorSelection
    from ..models.connector_binding_event import ConnectorBindingEvent
    from ..models.connector_binding_policy import ConnectorBindingPolicy
    from ..models.connector_tool_grant import ConnectorToolGrant


T = TypeVar("T", bound="AgentConnectorBinding")


@_attrs_define
class AgentConnectorBinding:
    """A connector whose tools an agent config may call, under an alias. The binding is the grant: only the tools it lists
    are offered, each pinned to the schema it was reviewed against, or for a tool a session binding grants by name
    alone, to the schema its connection first offered it with.

        Attributes:
            connection (AgentConnectorSelection): Which connection a binding's tools are called through.
            connector_id (str): A connector definition the app can see: a built-in, or one of its own, whose id starts with
                custom_.
            name (str): The alias, unique within the config: a lowercase letter, then up to 62 lowercase letters, digits, -
                or _, never __ and not ending in _. The model is offered each tool as <name>__<tool>, split back at the first
                __, so a __ inside the alias or a _ at its end would split it in the wrong place.
            tools (list[ConnectorToolGrant]): The exact tools allowed, each named once. There is no wildcard, and an empty
                list grants none. A session binding may grant a tool by name alone, which pins its schema per connection on
                first use.
            events (list[ConnectorBindingEvent] | Unset): MCP events the binding's fixed connection is subscribed to, each
                opening a text conversation from the config when it arrives. Subscribed when the connection is next validated.
                Only a fixed binding may declare events: a session binding's connection is picked when a session opens, and an
                event arrives with no session open.
            policy (ConnectorBindingPolicy | Unset): How a binding's tool calls behave around speech and interruptions.
                Every field is optional, and a field left out keeps today's behaviour.
            required (bool | Unset): Whether a session needs this connector. A required one that cannot be opened fails the
                session; an optional one is left out of it. Default: False.
            timeout_ms (int | Unset): How long one tool call may take, in milliseconds. Omitted, the session's default
                applies.
    """

    connection: AgentConnectorSelection
    connector_id: str
    name: str
    tools: list[ConnectorToolGrant]
    events: list[ConnectorBindingEvent] | Unset = UNSET
    policy: ConnectorBindingPolicy | Unset = UNSET
    required: bool | Unset = False
    timeout_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connection = self.connection.to_dict()

        connector_id = self.connector_id

        name = self.name

        tools = []
        for tools_item_data in self.tools:
            tools_item = tools_item_data.to_dict()
            tools.append(tools_item)

        events: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.events, Unset):
            events = []
            for events_item_data in self.events:
                events_item = events_item_data.to_dict()
                events.append(events_item)

        policy: dict[str, Any] | Unset = UNSET
        if not isinstance(self.policy, Unset):
            policy = self.policy.to_dict()

        required = self.required

        timeout_ms = self.timeout_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection": connection,
                "connector_id": connector_id,
                "name": name,
                "tools": tools,
            }
        )
        if events is not UNSET:
            field_dict["events"] = events
        if policy is not UNSET:
            field_dict["policy"] = policy
        if required is not UNSET:
            field_dict["required"] = required
        if timeout_ms is not UNSET:
            field_dict["timeout_ms"] = timeout_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_connector_selection import (
            AgentConnectorSelection,
        )
        from ..models.connector_binding_event import (
            ConnectorBindingEvent,
        )
        from ..models.connector_binding_policy import (
            ConnectorBindingPolicy,
        )
        from ..models.connector_tool_grant import ConnectorToolGrant

        d = dict(src_dict)
        connection = AgentConnectorSelection.from_dict(d.pop("connection"))

        connector_id = d.pop("connector_id")

        name = d.pop("name")

        tools = []
        _tools = d.pop("tools")
        for tools_item_data in _tools:
            tools_item = ConnectorToolGrant.from_dict(tools_item_data)

            tools.append(tools_item)

        _events = d.pop("events", UNSET)
        events: list[ConnectorBindingEvent] | Unset = UNSET
        if _events is not UNSET:
            events = []
            for events_item_data in _events:
                events_item = ConnectorBindingEvent.from_dict(events_item_data)

                events.append(events_item)

        _policy = d.pop("policy", UNSET)
        policy: ConnectorBindingPolicy | Unset
        if isinstance(_policy, Unset):
            policy = UNSET
        else:
            policy = ConnectorBindingPolicy.from_dict(_policy)

        required = d.pop("required", UNSET)

        timeout_ms = d.pop("timeout_ms", UNSET)

        agent_connector_binding = cls(
            connection=connection,
            connector_id=connector_id,
            name=name,
            tools=tools,
            events=events,
            policy=policy,
            required=required,
            timeout_ms=timeout_ms,
        )

        agent_connector_binding.additional_properties = d
        return agent_connector_binding

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
