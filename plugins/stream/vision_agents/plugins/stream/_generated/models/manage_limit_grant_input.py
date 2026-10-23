from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.limit_subject import LimitSubject
from ..types import UNSET, Unset

T = TypeVar("T", bound="ManageLimitGrantInput")


@_attrs_define
class ManageLimitGrantInput:
    """A one-time grant staff make for an organization, for support. It is the customer's like any other.

    Attributes:
        amount_micros (int):
        organization_id (str):
        reason (str): Why it is made; more than spaces.
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an end user of an app, a
            usage category across the organization or within one app, or a team across the organization or within one app. A
            team is the user's team, as its token names it.
        api_key (str | Unset): The key: 1 to 128 characters of A-Z, a-z, 0-9, _ and -.
        app_id (str | Unset): The app. Required at app, api_key and user level, and for an app's category or team.
        category (str | Unset): The usage category, for subject category or app_category: 1 to 64 characters of A-Z,
            a-z, 0-9, _ . : and -. A team is subject team, never a category starting team:.
        expires_at (datetime.datetime | Unset): When it ends, in the future and at most 366 days away. Left out, it ends
            when used up or revoked. The next midnight UTC makes it more for today.
        team (str | Unset): The team, for subject team or app_team: 1 to 59 characters of A-Z, a-z, 0-9, _ . and -.
        user_id (str | Unset): The end user, as the app's tokens name them: 1 to 255 characters of A-Z, a-z, 0-9, @, _
            and -.
    """

    amount_micros: int
    organization_id: str
    reason: str
    subject: LimitSubject
    api_key: str | Unset = UNSET
    app_id: str | Unset = UNSET
    category: str | Unset = UNSET
    expires_at: datetime.datetime | Unset = UNSET
    team: str | Unset = UNSET
    user_id: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        amount_micros = self.amount_micros

        organization_id = self.organization_id

        reason = self.reason

        subject = self.subject.value

        api_key = self.api_key

        app_id = self.app_id

        category = self.category

        expires_at: str | Unset = UNSET
        if not isinstance(self.expires_at, Unset):
            expires_at = self.expires_at.isoformat()

        team = self.team

        user_id = self.user_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "amount_micros": amount_micros,
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
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at
        if team is not UNSET:
            field_dict["team"] = team
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        amount_micros = d.pop("amount_micros")

        organization_id = d.pop("organization_id")

        reason = d.pop("reason")

        subject = LimitSubject(d.pop("subject"))

        api_key = d.pop("api_key", UNSET)

        app_id = d.pop("app_id", UNSET)

        category = d.pop("category", UNSET)

        _expires_at = d.pop("expires_at", UNSET)
        expires_at: datetime.datetime | Unset
        if isinstance(_expires_at, Unset):
            expires_at = UNSET
        else:
            expires_at = datetime.datetime.fromisoformat(_expires_at)

        team = d.pop("team", UNSET)

        user_id = d.pop("user_id", UNSET)

        manage_limit_grant_input = cls(
            amount_micros=amount_micros,
            organization_id=organization_id,
            reason=reason,
            subject=subject,
            api_key=api_key,
            app_id=app_id,
            category=category,
            expires_at=expires_at,
            team=team,
            user_id=user_id,
        )

        return manage_limit_grant_input
