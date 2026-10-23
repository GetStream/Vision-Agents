from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.limit_interval import LimitInterval
from ..models.limit_subject import LimitSubject
from ..types import UNSET, Unset

T = TypeVar("T", bound="ManageLimitResetInput")


@_attrs_define
class ManageLimitResetInput:
    """A reset staff make of one subject's spend for its limits, for support.

    Attributes:
        organization_id (str):
        reason (str):
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an end user of an app, a
            usage category across the organization or within one app, or a team across the organization or within one app. A
            team is the user's team, as its token names it.
        api_key (str | Unset): The key: 1 to 128 characters of A-Z, a-z, 0-9, _ and -.
        app_id (str | Unset): The app. Required at app, api_key and user level, and for an app's category or team.
        category (str | Unset): The usage category, for subject category or app_category: 1 to 64 characters of A-Z,
            a-z, 0-9, _ . : and -. A team is subject team, never a category starting team:.
        intervals (list[LimitInterval] | None | Unset): The windows to reset. Left out, every one.
        team (str | Unset): The team, for subject team or app_team: 1 to 59 characters of A-Z, a-z, 0-9, _ . and -.
        user_id (str | Unset): The end user, as the app's tokens name them: 1 to 255 characters of A-Z, a-z, 0-9, @, _
            and -.
    """

    organization_id: str
    reason: str
    subject: LimitSubject
    api_key: str | Unset = UNSET
    app_id: str | Unset = UNSET
    category: str | Unset = UNSET
    intervals: list[LimitInterval] | None | Unset = UNSET
    team: str | Unset = UNSET
    user_id: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        organization_id = self.organization_id

        reason = self.reason

        subject = self.subject.value

        api_key = self.api_key

        app_id = self.app_id

        category = self.category

        intervals: list[str] | None | Unset
        if isinstance(self.intervals, Unset):
            intervals = UNSET
        elif isinstance(self.intervals, list):
            intervals = []
            for intervals_type_0_item_data in self.intervals:
                intervals_type_0_item = intervals_type_0_item_data.value
                intervals.append(intervals_type_0_item)

        else:
            intervals = self.intervals

        team = self.team

        user_id = self.user_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "organization_id": organization_id,
                "reason": reason,
                "subject": subject,
            }
        )
        if api_key is not UNSET:
            field_dict["api_key"] = api_key
        if app_id is not UNSET:
            field_dict["app_id"] = app_id
        if category is not UNSET:
            field_dict["category"] = category
        if intervals is not UNSET:
            field_dict["intervals"] = intervals
        if team is not UNSET:
            field_dict["team"] = team
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        organization_id = d.pop("organization_id")

        reason = d.pop("reason")

        subject = LimitSubject(d.pop("subject"))

        api_key = d.pop("api_key", UNSET)

        app_id = d.pop("app_id", UNSET)

        category = d.pop("category", UNSET)

        def _parse_intervals(data: object) -> list[LimitInterval] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                intervals_type_0 = []
                _intervals_type_0 = data
                for intervals_type_0_item_data in _intervals_type_0:
                    intervals_type_0_item = LimitInterval(intervals_type_0_item_data)

                    intervals_type_0.append(intervals_type_0_item)

                return intervals_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[LimitInterval] | None | Unset, data)

        intervals = _parse_intervals(d.pop("intervals", UNSET))

        team = d.pop("team", UNSET)

        user_id = d.pop("user_id", UNSET)

        manage_limit_reset_input = cls(
            organization_id=organization_id,
            reason=reason,
            subject=subject,
            api_key=api_key,
            app_id=app_id,
            category=category,
            intervals=intervals,
            team=team,
            user_id=user_id,
        )

        return manage_limit_reset_input
