from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.scheduled_task_next import ScheduledTaskNext
    from ..models.scheduled_task_schedule import ScheduledTaskSchedule


T = TypeVar("T", bound="UpdateScheduledTaskRequest")


@_attrs_define
class UpdateScheduledTaskRequest:
    """
    Attributes:
        instructions (str | Unset): What the agent is told when the task comes due. Server-side only: a device sending
            it is refused.
        next_ (ScheduledTaskNext | Unset):
        schedule (ScheduledTaskSchedule | Unset):
        title (str | Unset): A short name the user will recognise.
    """

    instructions: str | Unset = UNSET
    next_: ScheduledTaskNext | Unset = UNSET
    schedule: ScheduledTaskSchedule | Unset = UNSET
    title: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        instructions = self.instructions

        next_: dict[str, Any] | Unset = UNSET
        if not isinstance(self.next_, Unset):
            next_ = self.next_.to_dict()

        schedule: dict[str, Any] | Unset = UNSET
        if not isinstance(self.schedule, Unset):
            schedule = self.schedule.to_dict()

        title = self.title

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if next_ is not UNSET:
            field_dict["next"] = next_
        if schedule is not UNSET:
            field_dict["schedule"] = schedule
        if title is not UNSET:
            field_dict["title"] = title

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.scheduled_task_next import ScheduledTaskNext
        from ..models.scheduled_task_schedule import ScheduledTaskSchedule

        d = dict(src_dict)
        instructions = d.pop("instructions", UNSET)

        _next_ = d.pop("next", UNSET)
        next_: ScheduledTaskNext | Unset
        if isinstance(_next_, Unset):
            next_ = UNSET
        else:
            next_ = ScheduledTaskNext.from_dict(_next_)

        _schedule = d.pop("schedule", UNSET)
        schedule: ScheduledTaskSchedule | Unset
        if isinstance(_schedule, Unset):
            schedule = UNSET
        else:
            schedule = ScheduledTaskSchedule.from_dict(_schedule)

        title = d.pop("title", UNSET)

        update_scheduled_task_request = cls(
            instructions=instructions,
            next_=next_,
            schedule=schedule,
            title=title,
        )

        update_scheduled_task_request.additional_properties = d
        return update_scheduled_task_request

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
