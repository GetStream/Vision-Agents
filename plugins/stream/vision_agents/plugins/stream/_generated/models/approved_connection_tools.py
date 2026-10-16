from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.approved_connection_tool import ApprovedConnectionTool


T = TypeVar("T", bound="ApprovedConnectionTools")


@_attrs_define
class ApprovedConnectionTools:
    """
    Attributes:
        connection_id (str):
        tools (list[ApprovedConnectionTool] | None): The tools approved, by name, sorted.
    """

    connection_id: str
    tools: list[ApprovedConnectionTool] | None
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

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection_id": connection_id,
                "tools": tools,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.approved_connection_tool import ApprovedConnectionTool

        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        def _parse_tools(data: object) -> list[ApprovedConnectionTool] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tools_type_0 = []
                _tools_type_0 = data
                for tools_type_0_item_data in _tools_type_0:
                    tools_type_0_item = ApprovedConnectionTool.from_dict(
                        tools_type_0_item_data
                    )

                    tools_type_0.append(tools_type_0_item)

                return tools_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ApprovedConnectionTool] | None, data)

        tools = _parse_tools(d.pop("tools"))

        approved_connection_tools = cls(
            connection_id=connection_id,
            tools=tools,
        )

        approved_connection_tools.additional_properties = d
        return approved_connection_tools

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
