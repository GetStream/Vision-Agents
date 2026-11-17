from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SpendHardLimit")


@_attrs_define
class SpendHardLimit:
    """
    Attributes:
        daily_micros (int | Unset): The most spent in one UTC day, in micros; 0 or absent is no limit.
        monthly_micros (int | Unset): The most spent in one calendar month (UTC), in micros; 0 or absent is no limit.
    """

    daily_micros: int | Unset = UNSET
    monthly_micros: int | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        daily_micros = self.daily_micros

        monthly_micros = self.monthly_micros

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if daily_micros is not UNSET:
            field_dict["daily_micros"] = daily_micros
        if monthly_micros is not UNSET:
            field_dict["monthly_micros"] = monthly_micros

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        daily_micros = d.pop("daily_micros", UNSET)

        monthly_micros = d.pop("monthly_micros", UNSET)

        spend_hard_limit = cls(
            daily_micros=daily_micros,
            monthly_micros=monthly_micros,
        )

        return spend_hard_limit
