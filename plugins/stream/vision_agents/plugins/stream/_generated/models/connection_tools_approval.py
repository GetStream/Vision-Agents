from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.connection_tool_approval import ConnectionToolApproval


T = TypeVar("T", bound="ConnectionToolsApproval")


@_attrs_define
class ConnectionToolsApproval:
    """The tools of a connection to approve at the schema the provider lists now. An unknown field is refused rather than
    ignored.

        Attributes:
            tools (list[ConnectionToolApproval] | None): The tools to approve, each named once. Up to 128.
    """

    tools: list[ConnectionToolApproval] | None

    def to_dict(self) -> dict[str, Any]:
        tools: list[dict[str, Any]] | None
        if isinstance(self.tools, list):
            tools = []
            for tools_type_0_item_data in self.tools:
                tools_type_0_item = tools_type_0_item_data.to_dict()
                tools.append(tools_type_0_item)

        else:
            tools = self.tools

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "tools": tools,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connection_tool_approval import ConnectionToolApproval

        d = dict(src_dict)

        def _parse_tools(data: object) -> list[ConnectionToolApproval] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tools_type_0 = []
                _tools_type_0 = data
                for tools_type_0_item_data in _tools_type_0:
                    tools_type_0_item = ConnectionToolApproval.from_dict(
                        tools_type_0_item_data
                    )

                    tools_type_0.append(tools_type_0_item)

                return tools_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ConnectionToolApproval] | None, data)

        tools = _parse_tools(d.pop("tools"))

        connection_tools_approval = cls(
            tools=tools,
        )

        return connection_tools_approval
