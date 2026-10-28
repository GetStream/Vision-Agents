from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_preview_item_price_factors import PricePreviewItemPriceFactors
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
        region (str): Where the work runs, such as us or eu: a letter and up to 31 of a-z, 0-9 and hyphens. Work is
            priced only by an entry of its own region, never another region's.
        after (PricedEntry | Unset):
        before (PricedEntry | Unset):
        byok (bool | Unset): Work done with the customer's own key, which the vendor bills the customer for: priced only
            by a BYOK entry, and a BYOK entry prices no other work. Left out of an input, false.
        needs_confirmation (bool | Unset):
        price_factors (PricePreviewItemPriceFactors | Unset): By meter, the factor its price moves by. Absent for a
            meter with no price before or after.
        unpriced (bool | Unset): It is priced before the change and billable with no price after.
        when (PricePreviewItemWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}.
            Absent is the item's default entry.
    """

    modality: str
    model: str
    provider: str
    region: str
    after: PricedEntry | Unset = UNSET
    before: PricedEntry | Unset = UNSET
    byok: bool | Unset = UNSET
    needs_confirmation: bool | Unset = UNSET
    price_factors: PricePreviewItemPriceFactors | Unset = UNSET
    unpriced: bool | Unset = UNSET
    when: PricePreviewItemWhen | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        modality = self.modality

        model = self.model

        provider = self.provider

        region = self.region

        after: dict[str, Any] | Unset = UNSET
        if not isinstance(self.after, Unset):
            after = self.after.to_dict()

        before: dict[str, Any] | Unset = UNSET
        if not isinstance(self.before, Unset):
            before = self.before.to_dict()

        byok = self.byok

        needs_confirmation = self.needs_confirmation

        price_factors: dict[str, Any] | Unset = UNSET
        if not isinstance(self.price_factors, Unset):
            price_factors = self.price_factors.to_dict()

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
                "region": region,
            }
        )
        if after is not UNSET:
            field_dict["after"] = after
        if before is not UNSET:
            field_dict["before"] = before
        if byok is not UNSET:
            field_dict["byok"] = byok
        if needs_confirmation is not UNSET:
            field_dict["needs_confirmation"] = needs_confirmation
        if price_factors is not UNSET:
            field_dict["price_factors"] = price_factors
        if unpriced is not UNSET:
            field_dict["unpriced"] = unpriced
        if when is not UNSET:
            field_dict["when"] = when

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_preview_item_price_factors import (
            PricePreviewItemPriceFactors,
        )
        from ..models.price_preview_item_when import PricePreviewItemWhen
        from ..models.priced_entry import PricedEntry

        d = dict(src_dict)
        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

        region = d.pop("region")

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

        byok = d.pop("byok", UNSET)

        needs_confirmation = d.pop("needs_confirmation", UNSET)

        _price_factors = d.pop("price_factors", UNSET)
        price_factors: PricePreviewItemPriceFactors | Unset
        if isinstance(_price_factors, Unset):
            price_factors = UNSET
        else:
            price_factors = PricePreviewItemPriceFactors.from_dict(_price_factors)

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
            region=region,
            after=after,
            before=before,
            byok=byok,
            needs_confirmation=needs_confirmation,
            price_factors=price_factors,
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
