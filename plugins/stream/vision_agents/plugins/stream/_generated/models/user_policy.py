from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.effective_quota import EffectiveQuota
    from ..models.quota import Quota


T = TypeVar("T", bound="UserPolicy")


@_attrs_define
class UserPolicy:
    """One end user of the calling app: their own quota, if any, the quota they are held to, and what they spent today.

    Attributes:
        effective (EffectiveQuota): The quota a user is held to: each number is the first set of the user's own, the
            app's, the organization's and the router's default. Both are 0 on a router that counts nothing.
        messages (int): Responses counted today toward a quota the customer set, the app's backend acting for the user
            included. A day held only to the router's default counts the user's devices alone, which this leaves out.
        resets_at (datetime.datetime): When the day ends and the count starts again: the next UTC midnight.
        tokens (int): Tokens counted today toward a quota the customer set, counted as messages is.
        user_id (str):
        quota (Quota | Unset): What one end user may spend in a UTC day. Each number is the first set of the user's own
            (/v1/policies/users/{user_id}), the app's, the organization's and the router's default. A quota set here counts
            everything recorded with the user: what their devices ask for and what the app's backend asks for in their name.
            (An app's quota counted devices only before; the router's default still does.) A device is counted by its
            address as well. A turn is counted on the day of the user who sent it. A request over the limit is a 429
            rate_limited with Retry-After, whose message names the quota (user <id>, every user of app <id>, every user of
            organization <id>), and a counted request carries X-Quota-Limit, X-Quota-Remaining and X-Quota-Reset.
    """

    effective: EffectiveQuota
    messages: int
    resets_at: datetime.datetime
    tokens: int
    user_id: str
    quota: Quota | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        effective = self.effective.to_dict()

        messages = self.messages

        resets_at = self.resets_at.isoformat()

        tokens = self.tokens

        user_id = self.user_id

        quota: dict[str, Any] | Unset = UNSET
        if not isinstance(self.quota, Unset):
            quota = self.quota.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "effective": effective,
                "messages": messages,
                "resets_at": resets_at,
                "tokens": tokens,
                "user_id": user_id,
            }
        )
        if quota is not UNSET:
            field_dict["quota"] = quota

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.effective_quota import EffectiveQuota
        from ..models.quota import Quota

        d = dict(src_dict)
        effective = EffectiveQuota.from_dict(d.pop("effective"))

        messages = d.pop("messages")

        resets_at = datetime.datetime.fromisoformat(d.pop("resets_at"))

        tokens = d.pop("tokens")

        user_id = d.pop("user_id")

        _quota = d.pop("quota", UNSET)
        quota: Quota | Unset
        if isinstance(_quota, Unset):
            quota = UNSET
        else:
            quota = Quota.from_dict(_quota)

        user_policy = cls(
            effective=effective,
            messages=messages,
            resets_at=resets_at,
            tokens=tokens,
            user_id=user_id,
            quota=quota,
        )

        user_policy.additional_properties = d
        return user_policy

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
