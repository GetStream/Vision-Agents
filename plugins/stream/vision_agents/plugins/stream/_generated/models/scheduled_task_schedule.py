from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ScheduledTaskSchedule")


@_attrs_define
class ScheduledTaskSchedule:
    """
    Attributes:
        at (str | Unset): When it first runs, as the user's clock reads it, 2026-10-20T09:00:00; with an offset or Z it
            is that instant, read in the schedule's zone. Exactly one of at and in is given on a write.
        description (str | Unset): What the schedule says, in one sentence. Read-only: written by the router, and the
            same sentence the model reads back. Example: Every weekday at 09:00 (Europe/Amsterdam).
        in_ (str | Unset): How long from now instead, an ISO 8601 duration: PT20M, P1D. Days are calendar days: P1D is
            the same time tomorrow. Resolved on write; a response never carries it. Example: PT20M.
        repeat (str | Unset): Omit for once. Otherwise an RRULE subset without DTSTART or UNTIL: FREQ=DAILY,
            FREQ=WEEKLY;BYDAY=MO,WE,FR, FREQ=MONTHLY;BYMONTHDAY=1. Example: FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR.
        timezone (str | Unset): An IANA zone, such as Europe/Amsterdam. On create, left out resolves to the user's
            stored zone, then the agent's default_timezone; on update, to the task's own zone. Example: Europe/Amsterdam.
        until (str | Unset): The last local date it repeats on, inclusive, 2026-12-31. Example: 2026-12-31.
    """

    at: str | Unset = UNSET
    description: str | Unset = UNSET
    in_: str | Unset = UNSET
    repeat: str | Unset = UNSET
    timezone: str | Unset = UNSET
    until: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        at = self.at

        description = self.description

        in_ = self.in_

        repeat = self.repeat

        timezone = self.timezone

        until = self.until

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if at is not UNSET:
            field_dict["at"] = at
        if description is not UNSET:
            field_dict["description"] = description
        if in_ is not UNSET:
            field_dict["in"] = in_
        if repeat is not UNSET:
            field_dict["repeat"] = repeat
        if timezone is not UNSET:
            field_dict["timezone"] = timezone
        if until is not UNSET:
            field_dict["until"] = until

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        at = d.pop("at", UNSET)

        description = d.pop("description", UNSET)

        in_ = d.pop("in", UNSET)

        repeat = d.pop("repeat", UNSET)

        timezone = d.pop("timezone", UNSET)

        until = d.pop("until", UNSET)

        scheduled_task_schedule = cls(
            at=at,
            description=description,
            in_=in_,
            repeat=repeat,
            timezone=timezone,
            until=until,
        )

        scheduled_task_schedule.additional_properties = d
        return scheduled_task_schedule

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
