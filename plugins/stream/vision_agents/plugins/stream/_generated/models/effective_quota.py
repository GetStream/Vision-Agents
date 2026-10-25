from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.quota_level import QuotaLevel

T = TypeVar("T", bound="EffectiveQuota")


@_attrs_define
class EffectiveQuota:
    """The quota a user is held to: each number is the first set of the user's own, the app's, the organization's and the
    router's default. Both are 0 on a router that counts nothing.

        Attributes:
            messages_per_user_per_day (int): Responses the user may ask for today. 0 is no cap.
            messages_set_by (QuotaLevel): Who set a number of a quota: the user's own policy, the app's, the organization's,
                or the router's default. The router's default counts the user's devices only; every other counts the app's
                backend acting for the user too.
            tokens_per_user_per_day (int): Tokens the user's responses may cost today. 0 is no cap.
            tokens_set_by (QuotaLevel): Who set a number of a quota: the user's own policy, the app's, the organization's,
                or the router's default. The router's default counts the user's devices only; every other counts the app's
                backend acting for the user too.
    """

    messages_per_user_per_day: int
    messages_set_by: QuotaLevel
    tokens_per_user_per_day: int
    tokens_set_by: QuotaLevel
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        messages_per_user_per_day = self.messages_per_user_per_day

        messages_set_by = self.messages_set_by.value

        tokens_per_user_per_day = self.tokens_per_user_per_day

        tokens_set_by = self.tokens_set_by.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "messages_per_user_per_day": messages_per_user_per_day,
                "messages_set_by": messages_set_by,
                "tokens_per_user_per_day": tokens_per_user_per_day,
                "tokens_set_by": tokens_set_by,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        messages_per_user_per_day = d.pop("messages_per_user_per_day")

        messages_set_by = QuotaLevel(d.pop("messages_set_by"))

        tokens_per_user_per_day = d.pop("tokens_per_user_per_day")

        tokens_set_by = QuotaLevel(d.pop("tokens_set_by"))

        effective_quota = cls(
            messages_per_user_per_day=messages_per_user_per_day,
            messages_set_by=messages_set_by,
            tokens_per_user_per_day=tokens_per_user_per_day,
            tokens_set_by=tokens_set_by,
        )

        effective_quota.additional_properties = d
        return effective_quota

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
