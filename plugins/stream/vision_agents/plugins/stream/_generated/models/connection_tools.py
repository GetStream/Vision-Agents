from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connection_tool import ConnectionTool


T = TypeVar("T", bound="ConnectionTools")


@_attrs_define
class ConnectionTools:
    """The tools a connection offered when it was last validated, in one piece: the provider's own list, not a page of one.

    Attributes:
        connection_id (str):
        tools (list[ConnectionTool] | None):
        checked_at (datetime.datetime | Unset): When the list was read. Absent until a validate listed it.
        digest (str | Unset): The digest of the whole list. Absent until a validate listed it.
    """

    connection_id: str
    tools: list[ConnectionTool] | None
    checked_at: datetime.datetime | Unset = UNSET
    digest: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connection_id = self.connection_id

        tools: list[dict[str, Any]] | None
        if isinstance(self.tools, list):
            tools = []
            for tools_type_0_item_data in self.tools:
                tools_type_0_item = tools_type_0_item_data.to_dict()
                tools.append(tools_type_0_item)

        else:
            tools = self.tools

        checked_at: str | Unset = UNSET
        if not isinstance(self.checked_at, Unset):
            checked_at = self.checked_at.isoformat()

        digest = self.digest

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection_id": connection_id,
                "tools": tools,
            }
        )
        if checked_at is not UNSET:
            field_dict["checked_at"] = checked_at
        if digest is not UNSET:
            field_dict["digest"] = digest

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connection_tool import ConnectionTool

        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        def _parse_tools(data: object) -> list[ConnectionTool] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tools_type_0 = []
                _tools_type_0 = data
                for tools_type_0_item_data in _tools_type_0:
                    tools_type_0_item = ConnectionTool.from_dict(tools_type_0_item_data)

                    tools_type_0.append(tools_type_0_item)

                return tools_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ConnectionTool] | None, data)

        tools = _parse_tools(d.pop("tools"))

        _checked_at = d.pop("checked_at", UNSET)
        checked_at: datetime.datetime | Unset
        if isinstance(_checked_at, Unset):
            checked_at = UNSET
        else:
            checked_at = datetime.datetime.fromisoformat(_checked_at)

        digest = d.pop("digest", UNSET)

        connection_tools = cls(
            connection_id=connection_id,
            tools=tools,
            checked_at=checked_at,
            digest=digest,
        )

        connection_tools.additional_properties = d
        return connection_tools

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
