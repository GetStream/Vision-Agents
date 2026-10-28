from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_change_entry_prices import PriceChangeEntryPrices
    from ..models.price_change_entry_rates import PriceChangeEntryRates
    from ..models.price_change_entry_when import PriceChangeEntryWhen
    from ..models.price_tier import PriceTier


T = TypeVar("T", bound="PriceChangeEntry")


@_attrs_define
class PriceChangeEntry:
    """
    Attributes:
        modality (str): Such as llm, stt, tts, sms or phone.
        model (str):
        provider (str):
        region (str): Where the work runs, such as us or eu: a letter and up to 31 of a-z, 0-9 and hyphens. Work is
            priced only by an entry of its own region, never another region's.
        sku (str): The item's SKU: the one it is priced under, or the one a withdrawal stopped pricing.
        byok (bool | Unset): Work done with the customer's own key, which the vendor bills the customer for: priced only
            by a BYOK entry, and a BYOK entry prices no other work. Left out of an input, false.
        free (bool | Unset):
        price_tiers (list[PriceTier] | None | Unset): What customers are charged for long prompts.
        prices (PriceChangeEntryPrices | Unset): What customers are charged, by meter.
        rates (PriceChangeEntryRates | Unset): The vendor's cost, by meter, for reference.
        tiers (list[PriceTier] | None | Unset): The vendor's cost for long prompts, for reference.
        when (PriceChangeEntryWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}.
            Absent is the item's default entry.
        withdrawn (bool | Unset): The change stopped pricing the item under its when.
    """

    modality: str
    model: str
    provider: str
    region: str
    sku: str
    byok: bool | Unset = UNSET
    free: bool | Unset = UNSET
    price_tiers: list[PriceTier] | None | Unset = UNSET
    prices: PriceChangeEntryPrices | Unset = UNSET
    rates: PriceChangeEntryRates | Unset = UNSET
    tiers: list[PriceTier] | None | Unset = UNSET
    when: PriceChangeEntryWhen | Unset = UNSET
    withdrawn: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        modality = self.modality

        model = self.model

        provider = self.provider

        region = self.region

        sku = self.sku

        byok = self.byok

        free = self.free

        price_tiers: list[dict[str, Any]] | None | Unset
        if isinstance(self.price_tiers, Unset):
            price_tiers = UNSET
        elif isinstance(self.price_tiers, list):
            price_tiers = []
            for price_tiers_type_0_item_data in self.price_tiers:
                price_tiers_type_0_item = price_tiers_type_0_item_data.to_dict()
                price_tiers.append(price_tiers_type_0_item)

        else:
            price_tiers = self.price_tiers

        prices: dict[str, Any] | Unset = UNSET
        if not isinstance(self.prices, Unset):
            prices = self.prices.to_dict()

        rates: dict[str, Any] | Unset = UNSET
        if not isinstance(self.rates, Unset):
            rates = self.rates.to_dict()

        tiers: list[dict[str, Any]] | None | Unset
        if isinstance(self.tiers, Unset):
            tiers = UNSET
        elif isinstance(self.tiers, list):
            tiers = []
            for tiers_type_0_item_data in self.tiers:
                tiers_type_0_item = tiers_type_0_item_data.to_dict()
                tiers.append(tiers_type_0_item)

        else:
            tiers = self.tiers

        when: dict[str, Any] | Unset = UNSET
        if not isinstance(self.when, Unset):
            when = self.when.to_dict()

        withdrawn = self.withdrawn

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "modality": modality,
                "model": model,
                "provider": provider,
                "region": region,
                "sku": sku,
            }
        )
        if byok is not UNSET:
            field_dict["byok"] = byok
        if free is not UNSET:
            field_dict["free"] = free
        if price_tiers is not UNSET:
            field_dict["price_tiers"] = price_tiers
        if prices is not UNSET:
            field_dict["prices"] = prices
        if rates is not UNSET:
            field_dict["rates"] = rates
        if tiers is not UNSET:
            field_dict["tiers"] = tiers
        if when is not UNSET:
            field_dict["when"] = when
        if withdrawn is not UNSET:
            field_dict["withdrawn"] = withdrawn

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_change_entry_prices import PriceChangeEntryPrices
        from ..models.price_change_entry_rates import PriceChangeEntryRates
        from ..models.price_change_entry_when import PriceChangeEntryWhen
        from ..models.price_tier import PriceTier

        d = dict(src_dict)
        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

        region = d.pop("region")

        sku = d.pop("sku")

        byok = d.pop("byok", UNSET)

        free = d.pop("free", UNSET)

        def _parse_price_tiers(data: object) -> list[PriceTier] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                price_tiers_type_0 = []
                _price_tiers_type_0 = data
                for price_tiers_type_0_item_data in _price_tiers_type_0:
                    price_tiers_type_0_item = PriceTier.from_dict(
                        price_tiers_type_0_item_data
                    )

                    price_tiers_type_0.append(price_tiers_type_0_item)

                return price_tiers_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[PriceTier] | None | Unset, data)

        price_tiers = _parse_price_tiers(d.pop("price_tiers", UNSET))

        _prices = d.pop("prices", UNSET)
        prices: PriceChangeEntryPrices | Unset
        if isinstance(_prices, Unset):
            prices = UNSET
        else:
            prices = PriceChangeEntryPrices.from_dict(_prices)

        _rates = d.pop("rates", UNSET)
        rates: PriceChangeEntryRates | Unset
        if isinstance(_rates, Unset):
            rates = UNSET
        else:
            rates = PriceChangeEntryRates.from_dict(_rates)

        def _parse_tiers(data: object) -> list[PriceTier] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tiers_type_0 = []
                _tiers_type_0 = data
                for tiers_type_0_item_data in _tiers_type_0:
                    tiers_type_0_item = PriceTier.from_dict(tiers_type_0_item_data)

                    tiers_type_0.append(tiers_type_0_item)

                return tiers_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[PriceTier] | None | Unset, data)

        tiers = _parse_tiers(d.pop("tiers", UNSET))

        _when = d.pop("when", UNSET)
        when: PriceChangeEntryWhen | Unset
        if isinstance(_when, Unset):
            when = UNSET
        else:
            when = PriceChangeEntryWhen.from_dict(_when)

        withdrawn = d.pop("withdrawn", UNSET)

        price_change_entry = cls(
            modality=modality,
            model=model,
            provider=provider,
            region=region,
            sku=sku,
            byok=byok,
            free=free,
            price_tiers=price_tiers,
            prices=prices,
            rates=rates,
            tiers=tiers,
            when=when,
            withdrawn=withdrawn,
        )

        price_change_entry.additional_properties = d
        return price_change_entry

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
