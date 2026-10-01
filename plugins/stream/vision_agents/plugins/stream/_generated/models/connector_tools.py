from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connector_tool import ConnectorTool


T = TypeVar("T", bound="ConnectorTools")


@_attrs_define
class ConnectorTools:
    """
    Attributes:
        connection_id (str):
        tools (list[ConnectorTool]):
        digest (str):
        checked_at (datetime.datetime | Unset):
    """

    connection_id: str
    tools: list[ConnectorTool]
    digest: str
    checked_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connection_id = self.connection_id

        tools = []
        for tools_item_data in self.tools:
            tools_item = tools_item_data.to_dict()
            tools.append(tools_item)

        digest = self.digest

        checked_at: str | Unset = UNSET
        if not isinstance(self.checked_at, Unset):
            checked_at = self.checked_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection_id": connection_id,
                "tools": tools,
                "digest": digest,
            }
        )
        if checked_at is not UNSET:
            field_dict["checked_at"] = checked_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connector_tool import ConnectorTool

        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        tools = []
        _tools = d.pop("tools")
        for tools_item_data in _tools:
            tools_item = ConnectorTool.from_dict(tools_item_data)

            tools.append(tools_item)

        digest = d.pop("digest")

        _checked_at = d.pop("checked_at", UNSET)
        checked_at: datetime.datetime | Unset
        if isinstance(_checked_at, Unset):
            checked_at = UNSET
        else:
            checked_at = datetime.datetime.fromisoformat(_checked_at)

        connector_tools = cls(
            connection_id=connection_id,
            tools=tools,
            digest=digest,
            checked_at=checked_at,
        )

        connector_tools.additional_properties = d
        return connector_tools

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
