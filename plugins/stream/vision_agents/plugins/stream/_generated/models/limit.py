from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.limit_interval import LimitInterval
from ..models.limit_level import LimitLevel
from ..models.limit_subject import LimitSubject
from ..types import UNSET, Unset

T = TypeVar("T", bound="Limit")


@_attrs_define
class Limit:
    """A spend limit: what one subject may spend in each window of its interval. Billable work for a subject with no room
    left under any of its limits is refused, 402 spend_limit_reached, or 429 user_limit_reached with Retry-After for an
    end user's. A subject is also held to every subject that contains it, so an app's work is refused once its
    organization's limit is spent.

        Attributes:
            created_at (datetime.datetime):
            effective_limit_micros (int): The lowest of limit_micros and the limits, in the same interval, of every subject
                that contains this one: what the subject can spend before something refuses it.
            id (int):
            interval (LimitInterval): The window a limit counts spend over, in UTC: the day, the week from Monday, 14 days
                counted from Monday 2024-01-01, or the calendar month.
            level (LimitLevel): Whose ids a limit names. A key's or a user's limit at organization or app level is a default
                for every key or user there, each counted on its own; at api_key or user level it names one. The most specific
                level wins. Every other subject has one level: the organization, a category and a team are organization level;
                an app, and a category or team within one, app level.
            limit_micros (int): What the subject may spend in each window, in millionths of a dollar.
            subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an end user of an app, a
                usage category across the organization or within one app, or a team across the organization or within one app. A
                team is the user's team, as its token names it.
            updated_at (datetime.datetime):
            updated_by (str): Who last wrote it, as the audit log names them.
            api_key (str | Unset): The key, for a limit at api_key level.
            app_id (str | Unset): The app, for a limit at app, api_key or user level, or on an app's category or team.
            category (str | Unset): The usage category, for a category's limit.
            note (str | Unset):
            team (str | Unset): The team, for a team's limit.
            user_id (str | Unset): The end user, for a limit at user level.
    """

    created_at: datetime.datetime
    effective_limit_micros: int
    id: int
    interval: LimitInterval
    level: LimitLevel
    limit_micros: int
    subject: LimitSubject
    updated_at: datetime.datetime
    updated_by: str
    api_key: str | Unset = UNSET
    app_id: str | Unset = UNSET
    category: str | Unset = UNSET
    note: str | Unset = UNSET
    team: str | Unset = UNSET
    user_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        effective_limit_micros = self.effective_limit_micros

        id = self.id

        interval = self.interval.value

        level = self.level.value

        limit_micros = self.limit_micros

        subject = self.subject.value

        updated_at = self.updated_at.isoformat()

        updated_by = self.updated_by

        api_key = self.api_key

        app_id = self.app_id

        category = self.category

        note = self.note

        team = self.team

        user_id = self.user_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "effective_limit_micros": effective_limit_micros,
                "id": id,
                "interval": interval,
                "level": level,
                "limit_micros": limit_micros,
                "subject": subject,
                "updated_at": updated_at,
                "updated_by": updated_by,
            }
        )
        if api_key is not UNSET:
            field_dict["api_key"] = api_key
        if app_id is not UNSET:
            field_dict["app_id"] = app_id
        if category is not UNSET:
            field_dict["category"] = category
        if note is not UNSET:
            field_dict["note"] = note
        if team is not UNSET:
            field_dict["team"] = team
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        effective_limit_micros = d.pop("effective_limit_micros")

        id = d.pop("id")

        interval = LimitInterval(d.pop("interval"))

        level = LimitLevel(d.pop("level"))

        limit_micros = d.pop("limit_micros")

        subject = LimitSubject(d.pop("subject"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        updated_by = d.pop("updated_by")

        api_key = d.pop("api_key", UNSET)

        app_id = d.pop("app_id", UNSET)

        category = d.pop("category", UNSET)

        note = d.pop("note", UNSET)

        team = d.pop("team", UNSET)

        user_id = d.pop("user_id", UNSET)

        limit = cls(
            created_at=created_at,
            effective_limit_micros=effective_limit_micros,
            id=id,
            interval=interval,
            level=level,
            limit_micros=limit_micros,
            subject=subject,
            updated_at=updated_at,
            updated_by=updated_by,
            api_key=api_key,
            app_id=app_id,
            category=category,
            note=note,
            team=team,
            user_id=user_id,
        )

        limit.additional_properties = d
        return limit

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
