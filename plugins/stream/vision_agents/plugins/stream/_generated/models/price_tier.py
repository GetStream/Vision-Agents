from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.price_tier_rates import PriceTierRates


T = TypeVar("T", bound="PriceTier")


@_attrs_define
class PriceTier:
    """
    Attributes:
        above_input_tokens (int): A prompt longer than this many input tokens is priced at these rates, for the whole
            request.
        rates (PriceTierRates): By meter.
    """

    above_input_tokens: int
    rates: PriceTierRates

    def to_dict(self) -> dict[str, Any]:
        above_input_tokens = self.above_input_tokens

        rates = self.rates.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "above_input_tokens": above_input_tokens,
                "rates": rates,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_tier_rates import PriceTierRates

        d = dict(src_dict)
        above_input_tokens = d.pop("above_input_tokens")

        rates = PriceTierRates.from_dict(d.pop("rates"))

        price_tier = cls(
            above_input_tokens=above_input_tokens,
            rates=rates,
        )

        return price_tier
