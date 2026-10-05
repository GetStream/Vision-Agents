from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_connector_selection import AgentConnectorSelection
    from ..models.connector_tool_grant import ConnectorToolGrant


T = TypeVar("T", bound="AgentConnectorBinding")


@_attrs_define
class AgentConnectorBinding:
    """A connector whose tools an agent config may call, under an alias. The binding is the grant: only the tools it lists
    are offered, each pinned to the schema it was reviewed against.

        Attributes:
            connection (AgentConnectorSelection): Which connection a binding's tools are called through.
            connector_id (str): A connector definition the app can see: a built-in, or one of its own, whose id starts with
                custom_.
            name (str): The alias, unique within the config: a lowercase letter, then up to 62 lowercase letters, digits, -
                or _, never __ and not ending in _. The model is offered each tool as <name>__<tool>, split back at the first
                __, so a __ inside the alias or a _ at its end would split it in the wrong place.
            tools (list[ConnectorToolGrant]): The exact tools allowed, each named once. There is no wildcard, and an empty
                list grants none.
            required (bool | Unset): Whether a session needs this connector. A required one that cannot be opened fails the
                session; an optional one is left out of it. Default: False.
            timeout_ms (int | Unset): How long one tool call may take, in milliseconds. Omitted, the session's default
                applies.
    """

    connection: AgentConnectorSelection
    connector_id: str
    name: str
    tools: list[ConnectorToolGrant]
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

        required = d.pop("required", UNSET)

        timeout_ms = d.pop("timeout_ms", UNSET)

        agent_connector_binding = cls(
            connection=connection,
            connector_id=connector_id,
            name=name,
            tools=tools,
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
