from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="Quota")


@_attrs_define
class Quota:
    """What one end user may spend in a UTC day. Each number is the first set of the user's own
    (/v1/policies/users/{user_id}), the app's, the organization's and the router's default. A quota set here counts
    everything recorded with the user: what their devices ask for and what the app's backend asks for in their name. (An
    app's quota counted devices only before; the router's default still does.) A device is counted by its address as
    well. A turn is counted on the day of the user who sent it. A request over the limit is a 429 rate_limited with
    Retry-After, whose message names the quota (user <id>, every user of app <id>, every user of organization <id>), and
    a counted request carries X-Quota-Limit, X-Quota-Remaining and X-Quota-Reset.

        Attributes:
            messages_per_user_per_day (int | Unset): How many responses one end user may ask for in a UTC day. 0 is no cap.
                Left out, the next level's applies: the app's, then the organization's, then the router's default. Example:
                20000.
            tokens_per_user_per_day (int | Unset): How many tokens, input and output, one end user's responses may cost in a
                UTC day. 0 is no cap. Left out, the next level's applies, as for messages. Example: 50000000.
    """

    messages_per_user_per_day: int | Unset = UNSET
    tokens_per_user_per_day: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        messages_per_user_per_day = self.messages_per_user_per_day

        tokens_per_user_per_day = self.tokens_per_user_per_day

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if messages_per_user_per_day is not UNSET:
            field_dict["messages_per_user_per_day"] = messages_per_user_per_day
        if tokens_per_user_per_day is not UNSET:
            field_dict["tokens_per_user_per_day"] = tokens_per_user_per_day

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        messages_per_user_per_day = d.pop("messages_per_user_per_day", UNSET)

        tokens_per_user_per_day = d.pop("tokens_per_user_per_day", UNSET)

        quota = cls(
            messages_per_user_per_day=messages_per_user_per_day,
            tokens_per_user_per_day=tokens_per_user_per_day,
        )

        quota.additional_properties = d
        return quota

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
