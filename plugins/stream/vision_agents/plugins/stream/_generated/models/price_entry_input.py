from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_entry_input_prices import PriceEntryInputPrices
    from ..models.price_entry_input_rates import PriceEntryInputRates
    from ..models.price_entry_input_when import PriceEntryInputWhen
    from ..models.price_tier import PriceTier


T = TypeVar("T", bound="PriceEntryInput")


@_attrs_define
class PriceEntryInput:
    """
    Attributes:
        modality (str): Such as llm, stt, tts, sms or phone.
        model (str):
        provider (str):
        free (bool | Unset): The vendor charges nothing for it. A free entry has no price.
        local (bool | Unset): Refused with invalid_price: a model is local when the routing config says so.
        price_tiers (list[PriceTier] | None | Unset): What customers are charged for long prompts: one tier for each of
            tiers, at its threshold and with its meters, or 400 price_tiers_mismatch.
        prices (PriceEntryInputPrices | Unset): What customers are charged, in the shape of rates: the same meters, or
            400 price_meters_mismatch. Required with rates; set outright, never worked out from them. Bounded as a rate is.
        rates (PriceEntryInputRates | Unset): The vendor's cost, by meter, kept for reference: input_tokens,
            cached_input_tokens, cache_write_input_tokens, output_tokens, characters, audio_ms, images, requests,
            message_segments, call_seconds and the rest pricing knows. Required unless the entry is free or withdrawn. An
            unknown meter, and a rate of more than $1000 a unit, are refused.
        sku (str | Unset): The item's SKU. Absent keeps the one the item has, or gives a new item one made from its
            modality, provider, model and when. A new item may take another item's SKU, as a model its vendor renamed keeps
            the old name's, only when that item is withdrawn in the same change or is no longer in force: else 400
            sku_in_use. An item in force keeps its SKU: another is 400 sku_changed.
        tiers (list[PriceTier] | None | Unset): The vendor's cost for long prompts. Each tier rates every meter the
            entry does.
        version (str | Unset): Refused with invalid_price: an entry's version is the change that writes it.
        when (PriceEntryInputWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}.
            Absent is the item's default entry.
        withdrawn (bool | Unset): Stop pricing the item under its when from effective_at on. Its entries under another
            when are untouched. A withdrawal carries no rate, price or SKU.
    """

    modality: str
    model: str
    provider: str
    free: bool | Unset = UNSET
    local: bool | Unset = UNSET
    price_tiers: list[PriceTier] | None | Unset = UNSET
    prices: PriceEntryInputPrices | Unset = UNSET
    rates: PriceEntryInputRates | Unset = UNSET
    sku: str | Unset = UNSET
    tiers: list[PriceTier] | None | Unset = UNSET
    version: str | Unset = UNSET
    when: PriceEntryInputWhen | Unset = UNSET
    withdrawn: bool | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        modality = self.modality

        model = self.model

        provider = self.provider

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

        withdrawn = self.withdrawn

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "modality": modality,
                "model": model,
                "provider": provider,
            }
        )
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
        if withdrawn is not UNSET:
            field_dict["withdrawn"] = withdrawn

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_entry_input_prices import PriceEntryInputPrices
        from ..models.price_entry_input_rates import PriceEntryInputRates
        from ..models.price_entry_input_when import PriceEntryInputWhen
        from ..models.price_tier import PriceTier

        d = dict(src_dict)
        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

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
        prices: PriceEntryInputPrices | Unset
        if isinstance(_prices, Unset):
            prices = UNSET
        else:
            prices = PriceEntryInputPrices.from_dict(_prices)

        _rates = d.pop("rates", UNSET)
        rates: PriceEntryInputRates | Unset
        if isinstance(_rates, Unset):
            rates = UNSET
        else:
            rates = PriceEntryInputRates.from_dict(_rates)

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
        when: PriceEntryInputWhen | Unset
        if isinstance(_when, Unset):
            when = UNSET
        else:
            when = PriceEntryInputWhen.from_dict(_when)

        withdrawn = d.pop("withdrawn", UNSET)

        price_entry_input = cls(
            modality=modality,
            model=model,
            provider=provider,
            free=free,
            local=local,
            price_tiers=price_tiers,
            prices=prices,
            rates=rates,
            sku=sku,
            tiers=tiers,
            version=version,
            when=when,
            withdrawn=withdrawn,
        )

        return price_entry_input
