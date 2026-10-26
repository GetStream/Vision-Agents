from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.quota import Quota


T = TypeVar("T", bound="UserPolicyInput")


@_attrs_define
class UserPolicyInput:
    """One end user's own policy, which holds a quota only.

    Attributes:
        quota (Quota | Unset): What one end user may spend in a UTC day. Each number is the first set of the user's own
            (/v1/policies/users/{user_id}), the app's, the organization's and the router's default. A quota set here counts
            everything recorded with the user: what their devices ask for and what the app's backend asks for in their name.
            (An app's quota counted devices only before; the router's default still does.) A device is counted by its
            address as well. A turn is counted on the day of the user who sent it. A request over the limit is a 429
            rate_limited with Retry-After, whose message names the quota (user <id>, every user of app <id>, every user of
            organization <id>), and a counted request carries X-Quota-Limit, X-Quota-Remaining and X-Quota-Reset.
    """

    quota: Quota | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        quota: dict[str, Any] | Unset = UNSET
        if not isinstance(self.quota, Unset):
            quota = self.quota.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if quota is not UNSET:
            field_dict["quota"] = quota

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.quota import Quota

        d = dict(src_dict)
        _quota = d.pop("quota", UNSET)
        quota: Quota | Unset
        if isinstance(_quota, Unset):
            quota = UNSET
        else:
            quota = Quota.from_dict(_quota)

        user_policy_input = cls(
            quota=quota,
        )

        return user_policy_input
