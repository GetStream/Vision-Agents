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
    """
    Attributes:
        name (str):
        connector_id (str):
        connection (AgentConnectorSelection):
        tools (list[ConnectorToolGrant]): Exact MCP tools allowed, each pinned to its reviewed schema digest. An empty
            list grants no tools.
        required (bool | Unset):  Default: False.
        timeout_ms (int | Unset):
    """

    name: str
    connector_id: str
    connection: AgentConnectorSelection
    tools: list[ConnectorToolGrant]
    required: bool | Unset = False
    timeout_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        connector_id = self.connector_id

        connection = self.connection.to_dict()

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
                "name": name,
                "connector_id": connector_id,
                "connection": connection,
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
        name = d.pop("name")

        connector_id = d.pop("connector_id")

        connection = AgentConnectorSelection.from_dict(d.pop("connection"))

        tools = []
        _tools = d.pop("tools")
        for tools_item_data in _tools:
            tools_item = ConnectorToolGrant.from_dict(tools_item_data)

            tools.append(tools_item)

        required = d.pop("required", UNSET)

        timeout_ms = d.pop("timeout_ms", UNSET)

        agent_connector_binding = cls(
            name=name,
            connector_id=connector_id,
            connection=connection,
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
