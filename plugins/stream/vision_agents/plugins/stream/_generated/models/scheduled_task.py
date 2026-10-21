from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.scheduled_task_created_by import ScheduledTaskCreatedBy
from ..models.scheduled_task_state import ScheduledTaskState
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.scheduled_task_custom import ScheduledTaskCustom
    from ..models.scheduled_task_quiet_hours import ScheduledTaskQuietHours
    from ..models.scheduled_task_schedule import ScheduledTaskSchedule


T = TypeVar("T", bound="ScheduledTask")


@_attrs_define
class ScheduledTask:
    """
    Attributes:
        config_id (str):
        consecutive_failures (int):
        created_at (datetime.datetime):
        created_by (ScheduledTaskCreatedBy): agent when the model made it in a conversation, api when a backend did.
        deliver_to (list[str]): Where an occurrence is delivered, in order, the first that lands.
        id (str):
        instructions (str):
        last_run_at (None | str): When a run last finished, in UTC. Null when none has.
        next_run_at (None | str): The next occurrence, RFC 3339 with the offset of the task's own zone. Null when there
            is none.
        schedule (ScheduledTaskSchedule):
        session_id (str): The conversation the task follows up in. Empty until its first occurrence opens one.
        state (ScheduledTaskState): active, paused, completed (a one-shot that ran, or a recurrence past its end), or
            failed (auto-paused after five failed occurrences in a row; resume clears it).
        title (str):
        upcoming (list[str]): Up to the next five occurrences, RFC 3339 with the task zone's offset.
        updated_at (datetime.datetime):
        user_id (str):
        custom (ScheduledTaskCustom | Unset):
        quiet_hours (ScheduledTaskQuietHours | Unset):
    """

    config_id: str
    consecutive_failures: int
    created_at: datetime.datetime
    created_by: ScheduledTaskCreatedBy
    deliver_to: list[str]
    id: str
    instructions: str
    last_run_at: None | str
    next_run_at: None | str
    schedule: ScheduledTaskSchedule
    session_id: str
    state: ScheduledTaskState
    title: str
    upcoming: list[str]
    updated_at: datetime.datetime
    user_id: str
    custom: ScheduledTaskCustom | Unset = UNSET
    quiet_hours: ScheduledTaskQuietHours | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        config_id = self.config_id

        consecutive_failures = self.consecutive_failures

        created_at = self.created_at.isoformat()

        created_by = self.created_by.value

        deliver_to = self.deliver_to

        id = self.id

        instructions = self.instructions

        last_run_at: None | str
        last_run_at = self.last_run_at

        next_run_at: None | str
        next_run_at = self.next_run_at

        schedule = self.schedule.to_dict()

        session_id = self.session_id

        state = self.state.value

        title = self.title

        upcoming = self.upcoming

        updated_at = self.updated_at.isoformat()

        user_id = self.user_id

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        quiet_hours: dict[str, Any] | Unset = UNSET
        if not isinstance(self.quiet_hours, Unset):
            quiet_hours = self.quiet_hours.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "config_id": config_id,
                "consecutive_failures": consecutive_failures,
                "created_at": created_at,
                "created_by": created_by,
                "deliver_to": deliver_to,
                "id": id,
                "instructions": instructions,
                "last_run_at": last_run_at,
                "next_run_at": next_run_at,
                "schedule": schedule,
                "session_id": session_id,
                "state": state,
                "title": title,
                "upcoming": upcoming,
                "updated_at": updated_at,
                "user_id": user_id,
            }
        )
        if custom is not UNSET:
            field_dict["custom"] = custom
        if quiet_hours is not UNSET:
            field_dict["quiet_hours"] = quiet_hours

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.scheduled_task_custom import ScheduledTaskCustom
        from ..models.scheduled_task_quiet_hours import ScheduledTaskQuietHours
        from ..models.scheduled_task_schedule import ScheduledTaskSchedule

        d = dict(src_dict)
        config_id = d.pop("config_id")

        consecutive_failures = d.pop("consecutive_failures")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        created_by = ScheduledTaskCreatedBy(d.pop("created_by"))

        deliver_to = cast(list[str], d.pop("deliver_to"))

        id = d.pop("id")

        instructions = d.pop("instructions")

        def _parse_last_run_at(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        last_run_at = _parse_last_run_at(d.pop("last_run_at"))

        def _parse_next_run_at(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        next_run_at = _parse_next_run_at(d.pop("next_run_at"))

        schedule = ScheduledTaskSchedule.from_dict(d.pop("schedule"))

        session_id = d.pop("session_id")

        state = ScheduledTaskState(d.pop("state"))

        title = d.pop("title")

        upcoming = cast(list[str], d.pop("upcoming"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        user_id = d.pop("user_id")

        _custom = d.pop("custom", UNSET)
        custom: ScheduledTaskCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = ScheduledTaskCustom.from_dict(_custom)

        _quiet_hours = d.pop("quiet_hours", UNSET)
        quiet_hours: ScheduledTaskQuietHours | Unset
        if isinstance(_quiet_hours, Unset):
            quiet_hours = UNSET
        else:
            quiet_hours = ScheduledTaskQuietHours.from_dict(_quiet_hours)

        scheduled_task = cls(
            config_id=config_id,
            consecutive_failures=consecutive_failures,
            created_at=created_at,
            created_by=created_by,
            deliver_to=deliver_to,
            id=id,
            instructions=instructions,
            last_run_at=last_run_at,
            next_run_at=next_run_at,
            schedule=schedule,
            session_id=session_id,
            state=state,
            title=title,
            upcoming=upcoming,
            updated_at=updated_at,
            user_id=user_id,
            custom=custom,
            quiet_hours=quiet_hours,
        )

        scheduled_task.additional_properties = d
        return scheduled_task

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
