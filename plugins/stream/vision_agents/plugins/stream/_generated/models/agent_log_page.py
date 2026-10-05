from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.agent_log import AgentLog


T = TypeVar("T", bound="AgentLogPage")


@_attrs_define
class AgentLogPage:
    """
    Attributes:
        coverage (str):
        dropped_logs (int):
        has_more (bool):
        items (list[AgentLog]):
        next_cursor (str):
        resume_cursor (str):
    """

    coverage: str
    dropped_logs: int
    has_more: bool
    items: list[AgentLog]
    next_cursor: str
    resume_cursor: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        coverage = self.coverage

        dropped_logs = self.dropped_logs

        has_more = self.has_more

        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        next_cursor = self.next_cursor

        resume_cursor = self.resume_cursor

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "coverage": coverage,
                "dropped_logs": dropped_logs,
                "has_more": has_more,
                "items": items,
                "next_cursor": next_cursor,
                "resume_cursor": resume_cursor,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_log import AgentLog

        d = dict(src_dict)
        coverage = d.pop("coverage")

        dropped_logs = d.pop("dropped_logs")

        has_more = d.pop("has_more")

        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = AgentLog.from_dict(items_item_data)

            items.append(items_item)

        next_cursor = d.pop("next_cursor")

        resume_cursor = d.pop("resume_cursor")

        agent_log_page = cls(
            coverage=coverage,
            dropped_logs=dropped_logs,
            has_more=has_more,
            items=items,
            next_cursor=next_cursor,
            resume_cursor=resume_cursor,
        )

        agent_log_page.additional_properties = d
        return agent_log_page

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
