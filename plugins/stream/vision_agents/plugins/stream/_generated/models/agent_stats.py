from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.tool_stats import ToolStats


T = TypeVar("T", bound="AgentStats")


@_attrs_define
class AgentStats:
    """
    Attributes:
        peak_at (datetime.datetime | None): When that peak was first reached. Null when no session was open in the
            window.
        peak_concurrent_sessions (int): The most sessions open at the same time, each one an agent running. A session
            still open counts up to now.
        tool_calls (ToolStats):
        tools (list[ToolStats]): One row per tool, most called first.
    """

    peak_at: datetime.datetime | None
    peak_concurrent_sessions: int
    tool_calls: ToolStats
    tools: list[ToolStats]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        peak_at: None | str
        if isinstance(self.peak_at, datetime.datetime):
            peak_at = self.peak_at.isoformat()
        else:
            peak_at = self.peak_at

        peak_concurrent_sessions = self.peak_concurrent_sessions

        tool_calls = self.tool_calls.to_dict()

        tools = []
        for tools_item_data in self.tools:
            tools_item = tools_item_data.to_dict()
            tools.append(tools_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "peak_at": peak_at,
                "peak_concurrent_sessions": peak_concurrent_sessions,
                "tool_calls": tool_calls,
                "tools": tools,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.tool_stats import ToolStats

        d = dict(src_dict)

        def _parse_peak_at(data: object) -> datetime.datetime | None:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                peak_at_type_0 = datetime.datetime.fromisoformat(data)

                return peak_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None, data)

        peak_at = _parse_peak_at(d.pop("peak_at"))

        peak_concurrent_sessions = d.pop("peak_concurrent_sessions")

        tool_calls = ToolStats.from_dict(d.pop("tool_calls"))

        tools = []
        _tools = d.pop("tools")
        for tools_item_data in _tools:
            tools_item = ToolStats.from_dict(tools_item_data)

            tools.append(tools_item)

        agent_stats = cls(
            peak_at=peak_at,
            peak_concurrent_sessions=peak_concurrent_sessions,
            tool_calls=tool_calls,
            tools=tools,
        )

        agent_stats.additional_properties = d
        return agent_stats

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
