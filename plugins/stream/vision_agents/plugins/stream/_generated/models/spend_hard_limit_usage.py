from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.spend_hard_limit_window import SpendHardLimitWindow


T = TypeVar("T", bound="SpendHardLimitUsage")


@_attrs_define
class SpendHardLimitUsage:
    """The most this organization may spend in the UTC day and in the calendar month, whatever its credit and its own
    limits. Set by the operator, never by the customer. A missing interval has no limit.

        Attributes:
            daily (SpendHardLimitWindow | Unset):
            monthly (SpendHardLimitWindow | Unset):
    """

    daily: SpendHardLimitWindow | Unset = UNSET
    monthly: SpendHardLimitWindow | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        daily: dict[str, Any] | Unset = UNSET
        if not isinstance(self.daily, Unset):
            daily = self.daily.to_dict()

        monthly: dict[str, Any] | Unset = UNSET
        if not isinstance(self.monthly, Unset):
            monthly = self.monthly.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if daily is not UNSET:
            field_dict["daily"] = daily
        if monthly is not UNSET:
            field_dict["monthly"] = monthly

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.spend_hard_limit_window import SpendHardLimitWindow

        d = dict(src_dict)
        _daily = d.pop("daily", UNSET)
        daily: SpendHardLimitWindow | Unset
        if isinstance(_daily, Unset):
            daily = UNSET
        else:
            daily = SpendHardLimitWindow.from_dict(_daily)

        _monthly = d.pop("monthly", UNSET)
        monthly: SpendHardLimitWindow | Unset
        if isinstance(_monthly, Unset):
            monthly = UNSET
        else:
            monthly = SpendHardLimitWindow.from_dict(_monthly)

        spend_hard_limit_usage = cls(
            daily=daily,
            monthly=monthly,
        )

        spend_hard_limit_usage.additional_properties = d
        return spend_hard_limit_usage

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
