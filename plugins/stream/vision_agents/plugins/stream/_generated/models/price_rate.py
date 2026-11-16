from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="PriceRate")


@_attrs_define
class PriceRate:
    """
    Attributes:
        per (int): The units the rate is for: 1000000 for a rate per million tokens, 1 for one per image.
        usd (float): US dollars for every per units.
        increment (int | Unset): The quantity is rounded up to a multiple of this before it is priced, such as 60 for a
            call billed by the started minute.
        minimum (int | Unset): The least quantity billed once anything is used.
    """

    per: int
    usd: float
    increment: int | Unset = UNSET
    minimum: int | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        per = self.per

        usd = self.usd

        increment = self.increment

        minimum = self.minimum

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "per": per,
                "usd": usd,
            }
        )
        if increment is not UNSET:
            field_dict["increment"] = increment
        if minimum is not UNSET:
            field_dict["minimum"] = minimum

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        per = d.pop("per")

        usd = d.pop("usd")

        increment = d.pop("increment", UNSET)

        minimum = d.pop("minimum", UNSET)

        price_rate = cls(
            per=per,
            usd=usd,
            increment=increment,
            minimum=minimum,
        )

        return price_rate
