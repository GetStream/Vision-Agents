from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.scheduled_task_next import ScheduledTaskNext
    from ..models.scheduled_task_quiet_hours import ScheduledTaskQuietHours
    from ..models.scheduled_task_schedule import ScheduledTaskSchedule


T = TypeVar("T", bound="UpdateScheduledTaskRequest")


@_attrs_define
class UpdateScheduledTaskRequest:
    """
    Attributes:
        deliver_to (list[str] | Unset): Replaces where an occurrence is delivered, in order. An empty list is the
            conversation alone.
        instructions (str | Unset): What the agent is told when the task comes due. Server-side only: a device sending
            it is refused.
        next_ (ScheduledTaskNext | Unset):
        quiet_hours (ScheduledTaskQuietHours | Unset):
        schedule (ScheduledTaskSchedule | Unset):
        title (str | Unset): A short name the user will recognise.
    """

    deliver_to: list[str] | Unset = UNSET
    instructions: str | Unset = UNSET
    next_: ScheduledTaskNext | Unset = UNSET
    quiet_hours: ScheduledTaskQuietHours | Unset = UNSET
    schedule: ScheduledTaskSchedule | Unset = UNSET
    title: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        deliver_to: list[str] | Unset = UNSET
        if not isinstance(self.deliver_to, Unset):
            deliver_to = self.deliver_to

        instructions = self.instructions

        next_: dict[str, Any] | Unset = UNSET
        if not isinstance(self.next_, Unset):
            next_ = self.next_.to_dict()

        quiet_hours: dict[str, Any] | Unset = UNSET
        if not isinstance(self.quiet_hours, Unset):
            quiet_hours = self.quiet_hours.to_dict()

        schedule: dict[str, Any] | Unset = UNSET
        if not isinstance(self.schedule, Unset):
            schedule = self.schedule.to_dict()

        title = self.title

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if deliver_to is not UNSET:
            field_dict["deliver_to"] = deliver_to
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if next_ is not UNSET:
            field_dict["next"] = next_
        if quiet_hours is not UNSET:
            field_dict["quiet_hours"] = quiet_hours
        if schedule is not UNSET:
            field_dict["schedule"] = schedule
        if title is not UNSET:
            field_dict["title"] = title

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.scheduled_task_next import ScheduledTaskNext
        from ..models.scheduled_task_quiet_hours import ScheduledTaskQuietHours
        from ..models.scheduled_task_schedule import ScheduledTaskSchedule

        d = dict(src_dict)
        deliver_to = cast(list[str], d.pop("deliver_to", UNSET))

        instructions = d.pop("instructions", UNSET)

        _next_ = d.pop("next", UNSET)
        next_: ScheduledTaskNext | Unset
        if isinstance(_next_, Unset):
            next_ = UNSET
        else:
            next_ = ScheduledTaskNext.from_dict(_next_)

        _quiet_hours = d.pop("quiet_hours", UNSET)
        quiet_hours: ScheduledTaskQuietHours | Unset
        if isinstance(_quiet_hours, Unset):
            quiet_hours = UNSET
        else:
            quiet_hours = ScheduledTaskQuietHours.from_dict(_quiet_hours)

        _schedule = d.pop("schedule", UNSET)
        schedule: ScheduledTaskSchedule | Unset
        if isinstance(_schedule, Unset):
            schedule = UNSET
        else:
            schedule = ScheduledTaskSchedule.from_dict(_schedule)

        title = d.pop("title", UNSET)

        update_scheduled_task_request = cls(
            deliver_to=deliver_to,
            instructions=instructions,
            next_=next_,
            quiet_hours=quiet_hours,
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
