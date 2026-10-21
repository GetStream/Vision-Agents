from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.create_scheduled_task_request_custom import (
        CreateScheduledTaskRequestCustom,
    )
    from ..models.scheduled_task_quiet_hours import ScheduledTaskQuietHours
    from ..models.scheduled_task_schedule import ScheduledTaskSchedule


T = TypeVar("T", bound="CreateScheduledTaskRequest")


@_attrs_define
class CreateScheduledTaskRequest:
    """
    Attributes:
        instructions (str): What the agent is told when the task comes due. Read back by the owner and by the customer's
            backend, so put nothing in it a user must not read.
        schedule (ScheduledTaskSchedule):
        title (str): A short name the user will recognise. Example: Dentist tomorrow.
        agent (str | Unset): The agent config by name, as session create names it. config_id or this, or session_id.
        config_id (str | Unset): The agent config by id. agent or this, or session_id.
        custom (CreateScheduledTaskRequestCustom | Unset): The caller's own labels, handed back untouched.
        deliver_to (list[str] | None | Unset): Where an occurrence is delivered, in order, the first that lands:
            conversation, sms, whatsapp, imessage or phone. The agent's reply is written in its conversation either way; a
            channel carries the same words out, and phone rings the user and speaks them. Left out or empty is the
            conversation alone.
        id (str | Unset): The task's id, chosen by the caller as a session id can be: up to 64 letters, digits, - and _.
            Left out, the router generates one. Sending one that is used answers 409. Example: booking-4821-reminder.
        quiet_hours (ScheduledTaskQuietHours | Unset):
        session_id (str | Unset): The named user's own session whose conversation the follow-up lands in; its config is
            the task's. Left out, the first occurrence opens a session.
    """

    instructions: str
    schedule: ScheduledTaskSchedule
    title: str
    agent: str | Unset = UNSET
    config_id: str | Unset = UNSET
    custom: CreateScheduledTaskRequestCustom | Unset = UNSET
    deliver_to: list[str] | None | Unset = UNSET
    id: str | Unset = UNSET
    quiet_hours: ScheduledTaskQuietHours | Unset = UNSET
    session_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        instructions = self.instructions

        schedule = self.schedule.to_dict()

        title = self.title

        agent = self.agent

        config_id = self.config_id

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        deliver_to: list[str] | None | Unset
        if isinstance(self.deliver_to, Unset):
            deliver_to = UNSET
        elif isinstance(self.deliver_to, list):
            deliver_to = self.deliver_to

        else:
            deliver_to = self.deliver_to

        id = self.id

        quiet_hours: dict[str, Any] | Unset = UNSET
        if not isinstance(self.quiet_hours, Unset):
            quiet_hours = self.quiet_hours.to_dict()

        session_id = self.session_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "instructions": instructions,
                "schedule": schedule,
                "title": title,
            }
        )
        if agent is not UNSET:
            field_dict["agent"] = agent
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if custom is not UNSET:
            field_dict["custom"] = custom
        if deliver_to is not UNSET:
            field_dict["deliver_to"] = deliver_to
        if id is not UNSET:
            field_dict["id"] = id
        if quiet_hours is not UNSET:
            field_dict["quiet_hours"] = quiet_hours
        if session_id is not UNSET:
            field_dict["session_id"] = session_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.create_scheduled_task_request_custom import (
            CreateScheduledTaskRequestCustom,
        )
        from ..models.scheduled_task_quiet_hours import ScheduledTaskQuietHours
        from ..models.scheduled_task_schedule import ScheduledTaskSchedule

        d = dict(src_dict)
        instructions = d.pop("instructions")

        schedule = ScheduledTaskSchedule.from_dict(d.pop("schedule"))

        title = d.pop("title")

        agent = d.pop("agent", UNSET)

        config_id = d.pop("config_id", UNSET)

        _custom = d.pop("custom", UNSET)
        custom: CreateScheduledTaskRequestCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = CreateScheduledTaskRequestCustom.from_dict(_custom)

        def _parse_deliver_to(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                deliver_to_type_0 = cast(list[str], data)

                return deliver_to_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        deliver_to = _parse_deliver_to(d.pop("deliver_to", UNSET))

        id = d.pop("id", UNSET)

        _quiet_hours = d.pop("quiet_hours", UNSET)
        quiet_hours: ScheduledTaskQuietHours | Unset
        if isinstance(_quiet_hours, Unset):
            quiet_hours = UNSET
        else:
            quiet_hours = ScheduledTaskQuietHours.from_dict(_quiet_hours)

        session_id = d.pop("session_id", UNSET)

        create_scheduled_task_request = cls(
            instructions=instructions,
            schedule=schedule,
            title=title,
            agent=agent,
            config_id=config_id,
            custom=custom,
            deliver_to=deliver_to,
            id=id,
            quiet_hours=quiet_hours,
            session_id=session_id,
        )

        create_scheduled_task_request.additional_properties = d
        return create_scheduled_task_request

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
