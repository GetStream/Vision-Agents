from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentScheduling")


@_attrs_define
class AgentScheduling:
    """Whether the agent may schedule tasks for its users, and the defaults those tasks take.

    Attributes:
        default_timezone (str | Unset): The IANA zone a scheduled task takes when neither the user nor their session has
            ever named one, such as America/New_York. A zone nobody knows is refused with a 400 invalid_timezone.
        enabled (bool | Unset): Offer the agent the scheduling tools (schedule_task, list_scheduled_tasks,
            update_scheduled_task, cancel_scheduled_task), so a user can ask it to remind them or do something later, once
            or on a repeat. Off by default: it adds four tools to every turn. Left out on an update, the stored setting
            stays.
        max_tasks_per_user (int | Unset): How many active scheduled tasks one user may hold. Omitted is 25.
    """

    default_timezone: str | Unset = UNSET
    enabled: bool | Unset = UNSET
    max_tasks_per_user: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        default_timezone = self.default_timezone

        enabled = self.enabled

        max_tasks_per_user = self.max_tasks_per_user

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if default_timezone is not UNSET:
            field_dict["default_timezone"] = default_timezone
        if enabled is not UNSET:
            field_dict["enabled"] = enabled
        if max_tasks_per_user is not UNSET:
            field_dict["max_tasks_per_user"] = max_tasks_per_user

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        default_timezone = d.pop("default_timezone", UNSET)

        enabled = d.pop("enabled", UNSET)

        max_tasks_per_user = d.pop("max_tasks_per_user", UNSET)

        agent_scheduling = cls(
            default_timezone=default_timezone,
            enabled=enabled,
            max_tasks_per_user=max_tasks_per_user,
        )

        agent_scheduling.additional_properties = d
        return agent_scheduling

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
