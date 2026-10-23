from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.limit_interval import LimitInterval
from ..models.limit_level import LimitLevel
from ..models.limit_subject import LimitSubject
from ..types import UNSET, Unset

T = TypeVar("T", bound="LimitInput")


@_attrs_define
class LimitInput:
    """A spend limit to create. Every limit is the customer's; an unknown field is refused.

    Attributes:
        interval (LimitInterval): The window a limit counts spend over, in UTC: the day, the week from Monday, 14 days
            counted from Monday 2024-01-01, or the calendar month.
        limit_micros (int): What the subject may spend in each window, in millionths of a dollar. 0 refuses every
            billable request.
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an end user of an app, a
            usage category across the organization or within one app, or a team across the organization or within one app. A
            team is the user's team, as its token names it.
        api_key (str | Unset): The key: 1 to 128 characters of A-Z, a-z, 0-9, _ and -.
        app_id (str | Unset): The app. Required at app, api_key and user level, and for an app's category or team.
        category (str | Unset): The usage category, for subject category or app_category: 1 to 64 characters of A-Z,
            a-z, 0-9, _ . : and -. A team is subject team, never a category starting team:.
        level (LimitLevel | Unset): Whose ids a limit names. A key's or a user's limit at organization or app level is a
            default for every key or user there, each counted on its own; at api_key or user level it names one. The most
            specific level wins. Every other subject has one level: the organization, a category and a team are organization
            level; an app, and a category or team within one, app level.
        note (str | Unset):
        team (str | Unset): The team, for subject team or app_team: 1 to 59 characters of A-Z, a-z, 0-9, _ . and -.
        user_id (str | Unset): The end user, as the app's tokens name them: 1 to 255 characters of A-Z, a-z, 0-9, @, _
            and -.
    """

    interval: LimitInterval
    limit_micros: int
    subject: LimitSubject
    api_key: str | Unset = UNSET
    app_id: str | Unset = UNSET
    category: str | Unset = UNSET
    level: LimitLevel | Unset = UNSET
    note: str | Unset = UNSET
    team: str | Unset = UNSET
    user_id: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        interval = self.interval.value

        limit_micros = self.limit_micros

        subject = self.subject.value

        api_key = self.api_key

        app_id = self.app_id

        category = self.category

        level: str | Unset = UNSET
        if not isinstance(self.level, Unset):
            level = self.level.value

        note = self.note

        team = self.team

        user_id = self.user_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "interval": interval,
                "limit_micros": limit_micros,
                "subject": subject,
            }
        )
        if api_key is not UNSET:
            field_dict["api_key"] = api_key
        if app_id is not UNSET:
            field_dict["app_id"] = app_id
        if category is not UNSET:
            field_dict["category"] = category
        if level is not UNSET:
            field_dict["level"] = level
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
        interval = LimitInterval(d.pop("interval"))

        limit_micros = d.pop("limit_micros")

        subject = LimitSubject(d.pop("subject"))

        api_key = d.pop("api_key", UNSET)

        app_id = d.pop("app_id", UNSET)

        category = d.pop("category", UNSET)

        _level = d.pop("level", UNSET)
        level: LimitLevel | Unset
        if isinstance(_level, Unset):
            level = UNSET
        else:
            level = LimitLevel(_level)

        note = d.pop("note", UNSET)

        team = d.pop("team", UNSET)

        user_id = d.pop("user_id", UNSET)

        limit_input = cls(
            interval=interval,
            limit_micros=limit_micros,
            subject=subject,
            api_key=api_key,
            app_id=app_id,
            category=category,
            level=level,
            note=note,
            team=team,
            user_id=user_id,
        )

        return limit_input
