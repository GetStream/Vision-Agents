from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_entry_prices import PriceEntryPrices
    from ..models.price_entry_rates import PriceEntryRates
    from ..models.price_entry_when import PriceEntryWhen
    from ..models.price_tier import PriceTier


T = TypeVar("T", bound="PriceEntry")


@_attrs_define
class PriceEntry:
    """
    Attributes:
        modality (str): Such as llm, stt, tts, sms or phone.
        model (str):
        provider (str):
        region (str): Where the work runs, such as us or eu: a letter and up to 31 of a-z, 0-9 and hyphens. Work is
            priced only by an entry of its own region, never another region's.
        byok (bool | Unset): Work done with the customer's own key, which the vendor bills the customer for: priced only
            by a BYOK entry, and a BYOK entry prices no other work. Left out of an input, false.
        effective_at (datetime.datetime | Unset): When the entry took effect, or takes effect.
        free (bool | Unset): The vendor charges nothing for it.
        local (bool | Unset): The deployment runs the model itself, as its routing config says. Local work with no price
            costs nothing.
        price_tiers (list[PriceTier] | None | Unset): What customers are charged for long prompts, at the thresholds of
            tiers.
        prices (PriceEntryPrices | Unset): What customers are charged, by the meters of rates.
        rates (PriceEntryRates | Unset): The vendor's cost, by meter, for reference: nothing is charged from it.
        sku (str | Unset): The item's SKU, the same across its price changes, which every row it prices records. Absent
            only on a local model no change prices.
        tiers (list[PriceTier] | None | Unset): The vendor's cost for long prompts, for reference.
        version (str | Unset): The price change that wrote the entry, which every row it prices names as its
            price_version.
        when (PriceEntryWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}.
            Absent is the item's default entry.
    """

    modality: str
    model: str
    provider: str
    region: str
    byok: bool | Unset = UNSET
    effective_at: datetime.datetime | Unset = UNSET
    free: bool | Unset = UNSET
    local: bool | Unset = UNSET
    price_tiers: list[PriceTier] | None | Unset = UNSET
    prices: PriceEntryPrices | Unset = UNSET
    rates: PriceEntryRates | Unset = UNSET
    sku: str | Unset = UNSET
    tiers: list[PriceTier] | None | Unset = UNSET
    version: str | Unset = UNSET
    when: PriceEntryWhen | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        modality = self.modality

        model = self.model

        provider = self.provider

        region = self.region

        byok = self.byok

        effective_at: str | Unset = UNSET
        if not isinstance(self.effective_at, Unset):
            effective_at = self.effective_at.isoformat()

        free = self.free

        local = self.local

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

        sku = self.sku

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

        version = self.version

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
        if byok is not UNSET:
            field_dict["byok"] = byok
        if effective_at is not UNSET:
            field_dict["effective_at"] = effective_at
        if free is not UNSET:
            field_dict["free"] = free
        if local is not UNSET:
            field_dict["local"] = local
        if price_tiers is not UNSET:
            field_dict["price_tiers"] = price_tiers
        if prices is not UNSET:
            field_dict["prices"] = prices
        if rates is not UNSET:
            field_dict["rates"] = rates
        if sku is not UNSET:
            field_dict["sku"] = sku
        if tiers is not UNSET:
            field_dict["tiers"] = tiers
        if version is not UNSET:
            field_dict["version"] = version
        if when is not UNSET:
            field_dict["when"] = when

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_entry_prices import PriceEntryPrices
        from ..models.price_entry_rates import PriceEntryRates
        from ..models.price_entry_when import PriceEntryWhen
        from ..models.price_tier import PriceTier

        d = dict(src_dict)
        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

        region = d.pop("region")

        byok = d.pop("byok", UNSET)

        _effective_at = d.pop("effective_at", UNSET)
        effective_at: datetime.datetime | Unset
        if isinstance(_effective_at, Unset):
            effective_at = UNSET
        else:
            effective_at = datetime.datetime.fromisoformat(_effective_at)

        free = d.pop("free", UNSET)

        local = d.pop("local", UNSET)

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
        prices: PriceEntryPrices | Unset
        if isinstance(_prices, Unset):
            prices = UNSET
        else:
            prices = PriceEntryPrices.from_dict(_prices)

        _rates = d.pop("rates", UNSET)
        rates: PriceEntryRates | Unset
        if isinstance(_rates, Unset):
            rates = UNSET
        else:
            rates = PriceEntryRates.from_dict(_rates)

        sku = d.pop("sku", UNSET)

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

        version = d.pop("version", UNSET)

        _when = d.pop("when", UNSET)
        when: PriceEntryWhen | Unset
        if isinstance(_when, Unset):
            when = UNSET
        else:
            when = PriceEntryWhen.from_dict(_when)

        price_entry = cls(
            modality=modality,
            model=model,
            provider=provider,
            region=region,
            byok=byok,
            effective_at=effective_at,
            free=free,
            local=local,
            price_tiers=price_tiers,
            prices=prices,
            rates=rates,
            sku=sku,
            tiers=tiers,
            version=version,
            when=when,
        )

        price_entry.additional_properties = d
        return price_entry

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
