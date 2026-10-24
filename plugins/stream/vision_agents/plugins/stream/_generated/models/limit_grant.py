from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.limit_grant_state import LimitGrantState
from ..models.limit_subject import LimitSubject
from ..types import UNSET, Unset

T = TypeVar("T", bound="LimitGrant")


@_attrs_define
class LimitGrant:
    """A one-time amount one subject may spend past its own limits once one of them is spent, until it is used up, expires
    or is revoked. It is not credit: the organization's balance still pays, and credit, the operator's spend hard limit
    and the limits of every subject that contains this one still refuse. Work admitted on it draws its whole cost from
    it, so the last request can take it a little past its amount.

        Attributes:
            amount_micros (int):
            created_at (datetime.datetime):
            created_by (str): Who made it, as the audit log names them.
            id (int):
            reason (str):
            remaining_micros (int): amount_micros less used_micros, never below 0.
            state (LimitGrantState): Where a grant stands: live (it is drawn once a limit of its subject is spent), used (it
                paid for its amount), expired (past expires_at) or revoked.
            subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an end user of an app, a
                usage category across the organization or within one app, or a team across the organization or within one app. A
                team is the user's team, as its token names it.
            used_micros (int): What the work admitted on it has cost: the whole billable cost of every row drawn from it, as
                the usage records have them.
            api_key (str | Unset):
            app_id (str | Unset):
            category (str | Unset):
            expires_at (datetime.datetime | Unset):
            revoked_at (datetime.datetime | Unset):
            revoked_by (str | Unset):
            team (str | Unset):
            user_id (str | Unset):
    """

    amount_micros: int
    created_at: datetime.datetime
    created_by: str
    id: int
    reason: str
    remaining_micros: int
    state: LimitGrantState
    subject: LimitSubject
    used_micros: int
    api_key: str | Unset = UNSET
    app_id: str | Unset = UNSET
    category: str | Unset = UNSET
    expires_at: datetime.datetime | Unset = UNSET
    revoked_at: datetime.datetime | Unset = UNSET
    revoked_by: str | Unset = UNSET
    team: str | Unset = UNSET
    user_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        amount_micros = self.amount_micros

        created_at = self.created_at.isoformat()

        created_by = self.created_by

        id = self.id

        reason = self.reason

        remaining_micros = self.remaining_micros

        state = self.state.value

        subject = self.subject.value

        used_micros = self.used_micros

        api_key = self.api_key

        app_id = self.app_id

        category = self.category

        expires_at: str | Unset = UNSET
        if not isinstance(self.expires_at, Unset):
            expires_at = self.expires_at.isoformat()

        revoked_at: str | Unset = UNSET
        if not isinstance(self.revoked_at, Unset):
            revoked_at = self.revoked_at.isoformat()

        revoked_by = self.revoked_by

        team = self.team

        user_id = self.user_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "amount_micros": amount_micros,
                "created_at": created_at,
                "created_by": created_by,
                "id": id,
                "reason": reason,
                "remaining_micros": remaining_micros,
                "state": state,
                "subject": subject,
                "used_micros": used_micros,
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
        if revoked_at is not UNSET:
            field_dict["revoked_at"] = revoked_at
        if revoked_by is not UNSET:
            field_dict["revoked_by"] = revoked_by
        if team is not UNSET:
            field_dict["team"] = team
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        amount_micros = d.pop("amount_micros")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        created_by = d.pop("created_by")

        id = d.pop("id")

        reason = d.pop("reason")

        remaining_micros = d.pop("remaining_micros")

        state = LimitGrantState(d.pop("state"))

        subject = LimitSubject(d.pop("subject"))

        used_micros = d.pop("used_micros")

        api_key = d.pop("api_key", UNSET)

        app_id = d.pop("app_id", UNSET)

        category = d.pop("category", UNSET)

        _expires_at = d.pop("expires_at", UNSET)
        expires_at: datetime.datetime | Unset
        if isinstance(_expires_at, Unset):
            expires_at = UNSET
        else:
            expires_at = datetime.datetime.fromisoformat(_expires_at)

        _revoked_at = d.pop("revoked_at", UNSET)
        revoked_at: datetime.datetime | Unset
        if isinstance(_revoked_at, Unset):
            revoked_at = UNSET
        else:
            revoked_at = datetime.datetime.fromisoformat(_revoked_at)

        revoked_by = d.pop("revoked_by", UNSET)

        team = d.pop("team", UNSET)

        user_id = d.pop("user_id", UNSET)

        limit_grant = cls(
            amount_micros=amount_micros,
            created_at=created_at,
            created_by=created_by,
            id=id,
            reason=reason,
            remaining_micros=remaining_micros,
            state=state,
            subject=subject,
            used_micros=used_micros,
            api_key=api_key,
            app_id=app_id,
            category=category,
            expires_at=expires_at,
            revoked_at=revoked_at,
            revoked_by=revoked_by,
            team=team,
            user_id=user_id,
        )

        limit_grant.additional_properties = d
        return limit_grant

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
