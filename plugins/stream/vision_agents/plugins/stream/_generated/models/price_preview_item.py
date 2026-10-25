from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_preview_item_rate_factors import PricePreviewItemRateFactors
    from ..models.price_preview_item_when import PricePreviewItemWhen
    from ..models.priced_entry import PricedEntry


T = TypeVar("T", bound="PricePreviewItem")


@_attrs_define
class PricePreviewItem:
    """
    Attributes:
        modality (str): Such as llm, stt, tts, sms or phone.
        model (str):
        provider (str):
        after (PricedEntry | Unset):
        before (PricedEntry | Unset):
        markup_factor (float | Unset): The factor the markup it is sold at moves by. Absent when either is 0.
        needs_confirmation (bool | Unset):
        rate_factors (PricePreviewItemRateFactors | Unset): By meter, the factor its vendor rate moves by. Absent for a
            meter with no rate before or after.
        unpriced (bool | Unset): It is priced before the change and billable with no rate after.
        when (PricePreviewItemWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}.
            Absent is the item's default entry.
    """

    modality: str
    model: str
    provider: str
    after: PricedEntry | Unset = UNSET
    before: PricedEntry | Unset = UNSET
    markup_factor: float | Unset = UNSET
    needs_confirmation: bool | Unset = UNSET
    rate_factors: PricePreviewItemRateFactors | Unset = UNSET
    unpriced: bool | Unset = UNSET
    when: PricePreviewItemWhen | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        modality = self.modality

        model = self.model

        provider = self.provider

        after: dict[str, Any] | Unset = UNSET
        if not isinstance(self.after, Unset):
            after = self.after.to_dict()

        before: dict[str, Any] | Unset = UNSET
        if not isinstance(self.before, Unset):
            before = self.before.to_dict()

        markup_factor = self.markup_factor

        needs_confirmation = self.needs_confirmation

        rate_factors: dict[str, Any] | Unset = UNSET
        if not isinstance(self.rate_factors, Unset):
            rate_factors = self.rate_factors.to_dict()

        unpriced = self.unpriced

        when: dict[str, Any] | Unset = UNSET
        if not isinstance(self.when, Unset):
            when = self.when.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "modality": modality,
                "model": model,
                "provider": provider,
            }
        )
        if after is not UNSET:
            field_dict["after"] = after
        if before is not UNSET:
            field_dict["before"] = before
        if markup_factor is not UNSET:
            field_dict["markup_factor"] = markup_factor
        if needs_confirmation is not UNSET:
            field_dict["needs_confirmation"] = needs_confirmation
        if rate_factors is not UNSET:
            field_dict["rate_factors"] = rate_factors
        if unpriced is not UNSET:
            field_dict["unpriced"] = unpriced
        if when is not UNSET:
            field_dict["when"] = when

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_preview_item_rate_factors import PricePreviewItemRateFactors
        from ..models.price_preview_item_when import PricePreviewItemWhen
        from ..models.priced_entry import PricedEntry

        d = dict(src_dict)
        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

        _after = d.pop("after", UNSET)
        after: PricedEntry | Unset
        if isinstance(_after, Unset):
            after = UNSET
        else:
            after = PricedEntry.from_dict(_after)

        _before = d.pop("before", UNSET)
        before: PricedEntry | Unset
        if isinstance(_before, Unset):
            before = UNSET
        else:
            before = PricedEntry.from_dict(_before)

        markup_factor = d.pop("markup_factor", UNSET)

        needs_confirmation = d.pop("needs_confirmation", UNSET)

        _rate_factors = d.pop("rate_factors", UNSET)
        rate_factors: PricePreviewItemRateFactors | Unset
        if isinstance(_rate_factors, Unset):
            rate_factors = UNSET
        else:
            rate_factors = PricePreviewItemRateFactors.from_dict(_rate_factors)

        unpriced = d.pop("unpriced", UNSET)

        _when = d.pop("when", UNSET)
        when: PricePreviewItemWhen | Unset
        if isinstance(_when, Unset):
            when = UNSET
        else:
            when = PricePreviewItemWhen.from_dict(_when)

        price_preview_item = cls(
            modality=modality,
            model=model,
            provider=provider,
            after=after,
            before=before,
            markup_factor=markup_factor,
            needs_confirmation=needs_confirmation,
            rate_factors=rate_factors,
            unpriced=unpriced,
            when=when,
        )

        price_preview_item.additional_properties = d
        return price_preview_item

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
