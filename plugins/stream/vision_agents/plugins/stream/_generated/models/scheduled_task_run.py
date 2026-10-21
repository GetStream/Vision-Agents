from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.scheduled_task_run_state import ScheduledTaskRunState
from ..models.scheduled_task_run_trigger import ScheduledTaskRunTrigger

T = TypeVar("T", bound="ScheduledTaskRun")


@_attrs_define
class ScheduledTaskRun:
    """
    Attributes:
        attempts (int):
        delivered_to (str): Where the follow-up landed: conversation, sms, whatsapp, imessage or phone. Empty until it
            lands.
        error (None | str): Why it failed, as a short code, when it did: delivery_failed, timed_out,
            schedule_unreadable, user_on_call, channel_unreachable, no_reply, reply_failed, no_phone_number,
            no_agent_number, not_answered or already_called (a call was placed for the occurrence on an earlier attempt, so
            the user is not rung again). The full error is in the router's logs. Null otherwise.
        finished_at (datetime.datetime | None):
        id (str):
        scheduled_for (str): When the occurrence was due, RFC 3339 with the task zone's offset.
        session_id (str):
        started_at (datetime.datetime | None):
        state (ScheduledTaskRunState): running (retries included), delivered, failed, or missed (it fell outside the
            grace period, or its call would have rung more than 15 minutes late).
        task_id (str):
        trigger (ScheduledTaskRunTrigger): schedule when it came due, manual when it was run by hand.
    """

    attempts: int
    delivered_to: str
    error: None | str
    finished_at: datetime.datetime | None
    id: str
    scheduled_for: str
    session_id: str
    started_at: datetime.datetime | None
    state: ScheduledTaskRunState
    task_id: str
    trigger: ScheduledTaskRunTrigger
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        attempts = self.attempts

        delivered_to = self.delivered_to

        error: None | str
        error = self.error

        finished_at: None | str
        if isinstance(self.finished_at, datetime.datetime):
            finished_at = self.finished_at.isoformat()
        else:
            finished_at = self.finished_at

        id = self.id

        scheduled_for = self.scheduled_for

        session_id = self.session_id

        started_at: None | str
        if isinstance(self.started_at, datetime.datetime):
            started_at = self.started_at.isoformat()
        else:
            started_at = self.started_at

        state = self.state.value

        task_id = self.task_id

        trigger = self.trigger.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "attempts": attempts,
                "delivered_to": delivered_to,
                "error": error,
                "finished_at": finished_at,
                "id": id,
                "scheduled_for": scheduled_for,
                "session_id": session_id,
                "started_at": started_at,
                "state": state,
                "task_id": task_id,
                "trigger": trigger,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        attempts = d.pop("attempts")

        delivered_to = d.pop("delivered_to")

        def _parse_error(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        error = _parse_error(d.pop("error"))

        def _parse_finished_at(data: object) -> datetime.datetime | None:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                finished_at_type_0 = datetime.datetime.fromisoformat(data)

                return finished_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None, data)

        finished_at = _parse_finished_at(d.pop("finished_at"))

        id = d.pop("id")

        scheduled_for = d.pop("scheduled_for")

        session_id = d.pop("session_id")

        def _parse_started_at(data: object) -> datetime.datetime | None:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                started_at_type_0 = datetime.datetime.fromisoformat(data)

                return started_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None, data)

        started_at = _parse_started_at(d.pop("started_at"))

        state = ScheduledTaskRunState(d.pop("state"))

        task_id = d.pop("task_id")

        trigger = ScheduledTaskRunTrigger(d.pop("trigger"))

        scheduled_task_run = cls(
            attempts=attempts,
            delivered_to=delivered_to,
            error=error,
            finished_at=finished_at,
            id=id,
            scheduled_for=scheduled_for,
            session_id=session_id,
            started_at=started_at,
            state=state,
            task_id=task_id,
            trigger=trigger,
        )

        scheduled_task_run.additional_properties = d
        return scheduled_task_run

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
