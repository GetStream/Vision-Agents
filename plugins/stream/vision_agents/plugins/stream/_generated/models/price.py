from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_rates import PriceRates
    from ..models.price_tier import PriceTier
    from ..models.price_when import PriceWhen


T = TypeVar("T", bound="Price")


@_attrs_define
class Price:
    """What one item costs: a model of a provider in a modality, in a region, with the customer's own key (byok) or
    without, under the attributes in when, at the rates every row of its work is debited at, which X-Stream-Cost-Micros
    reports. A byok price is a fee: the vendor bills the customer for the work itself.

        Attributes:
            free (bool): The work costs nothing.
            local (bool): The deployment runs the model itself; with no rates it costs nothing.
            modality (str): Such as llm, stt, tts, sms or phone.
            model (str):
            provider (str):
            region (str): Where the work runs, such as us or eu: a letter and up to 31 of a-z, 0-9 and hyphens. Work is
                priced only by an entry of its own region, never another region's.
            byok (bool | Unset): Work done with the customer's own key, which the vendor bills the customer for: priced only
                by a BYOK entry, and a BYOK entry prices no other work. Left out of an input, false.
            change (str | Unset): The price change that set it: on an entry in force, the price_version of every row it
                prices. Absent for a model the deployment runs itself that no change prices.
            description (str | Unset): In upcoming: why the change is made.
            effective_at (datetime.datetime | Unset): When it took effect, or takes effect. Absent when no change set it.
            rates (PriceRates | Unset): What a customer is charged, by meter.
            sku (str | Unset): The item's SKU, the same across its price changes, which every usage event of its work
                records. Absent for a model the deployment runs itself that no change prices.
            tiers (list[PriceTier] | None | Unset): What a customer is charged for long prompts.
            when (PriceWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}. Absent is
                the item's default entry.
            withdrawn (bool | Unset): In upcoming: the item stops being priced at effective_at.
    """

    free: bool
    local: bool
    modality: str
    model: str
    provider: str
    region: str
    byok: bool | Unset = UNSET
    change: str | Unset = UNSET
    description: str | Unset = UNSET
    effective_at: datetime.datetime | Unset = UNSET
    rates: PriceRates | Unset = UNSET
    sku: str | Unset = UNSET
    tiers: list[PriceTier] | None | Unset = UNSET
    when: PriceWhen | Unset = UNSET
    withdrawn: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        free = self.free

        local = self.local

        modality = self.modality

        model = self.model

        provider = self.provider

        region = self.region

        byok = self.byok

        change = self.change

        description = self.description

        effective_at: str | Unset = UNSET
        if not isinstance(self.effective_at, Unset):
            effective_at = self.effective_at.isoformat()

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

        when: dict[str, Any] | Unset = UNSET
        if not isinstance(self.when, Unset):
            when = self.when.to_dict()

        withdrawn = self.withdrawn

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "free": free,
                "local": local,
                "modality": modality,
                "model": model,
                "provider": provider,
                "region": region,
            }
        )
        if byok is not UNSET:
            field_dict["byok"] = byok
        if change is not UNSET:
            field_dict["change"] = change
        if description is not UNSET:
            field_dict["description"] = description
        if effective_at is not UNSET:
            field_dict["effective_at"] = effective_at
        if rates is not UNSET:
            field_dict["rates"] = rates
        if sku is not UNSET:
            field_dict["sku"] = sku
        if tiers is not UNSET:
            field_dict["tiers"] = tiers
        if when is not UNSET:
            field_dict["when"] = when
        if withdrawn is not UNSET:
            field_dict["withdrawn"] = withdrawn

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_rates import PriceRates
        from ..models.price_tier import PriceTier
        from ..models.price_when import PriceWhen

        d = dict(src_dict)
        free = d.pop("free")

        local = d.pop("local")

        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

        region = d.pop("region")

        byok = d.pop("byok", UNSET)

        change = d.pop("change", UNSET)

        description = d.pop("description", UNSET)

        _effective_at = d.pop("effective_at", UNSET)
        effective_at: datetime.datetime | Unset
        if isinstance(_effective_at, Unset):
            effective_at = UNSET
        else:
            effective_at = datetime.datetime.fromisoformat(_effective_at)

        _rates = d.pop("rates", UNSET)
        rates: PriceRates | Unset
        if isinstance(_rates, Unset):
            rates = UNSET
        else:
            rates = PriceRates.from_dict(_rates)

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

        _when = d.pop("when", UNSET)
        when: PriceWhen | Unset
        if isinstance(_when, Unset):
            when = UNSET
        else:
            when = PriceWhen.from_dict(_when)

        withdrawn = d.pop("withdrawn", UNSET)

        price = cls(
            free=free,
            local=local,
            modality=modality,
            model=model,
            provider=provider,
            region=region,
            byok=byok,
            change=change,
            description=description,
            effective_at=effective_at,
            rates=rates,
            sku=sku,
            tiers=tiers,
            when=when,
            withdrawn=withdrawn,
        )

        price.additional_properties = d
        return price

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
