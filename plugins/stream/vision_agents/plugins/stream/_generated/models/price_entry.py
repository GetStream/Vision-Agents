from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_entry_charged_rates import PriceEntryChargedRates
    from ..models.price_entry_rates import PriceEntryRates
    from ..models.price_entry_when import PriceEntryWhen
    from ..models.price_tier import PriceTier


T = TypeVar("T", bound="PriceEntry")


@_attrs_define
class PriceEntry:
    """
    Attributes:
        charged_markup_bps (int): The markup the entry is sold at: its own, else the list's.
        modality (str): Such as llm, stt, tts, sms or phone.
        model (str):
        provider (str):
        charged_rates (PriceEntryChargedRates | Unset): What customers are charged: each rate with the markup added, to
            12 significant digits.
        charged_tiers (list[PriceTier] | None | Unset): What customers are charged for long prompts.
        effective_at (datetime.datetime | Unset): When the entry took effect, or takes effect.
        free (bool | Unset): The vendor charges nothing for it.
        local (bool | Unset): The deployment runs the model itself, as its routing config says. Local work with no rate
            costs nothing.
        markup_bps (int | Unset): The entry's own markup in basis points, replacing the list's. Absent takes the list's.
        rates (PriceEntryRates | Unset): The vendor's rates, by meter.
        tiers (list[PriceTier] | None | Unset): The vendor's rates for long prompts.
        version (str | Unset): The price change that wrote the entry, which every row it prices names as its
            price_version.
        when (PriceEntryWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}.
            Absent is the item's default entry.
    """

    charged_markup_bps: int
    modality: str
    model: str
    provider: str
    charged_rates: PriceEntryChargedRates | Unset = UNSET
    charged_tiers: list[PriceTier] | None | Unset = UNSET
    effective_at: datetime.datetime | Unset = UNSET
    free: bool | Unset = UNSET
    local: bool | Unset = UNSET
    markup_bps: int | Unset = UNSET
    rates: PriceEntryRates | Unset = UNSET
    tiers: list[PriceTier] | None | Unset = UNSET
    version: str | Unset = UNSET
    when: PriceEntryWhen | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        charged_markup_bps = self.charged_markup_bps

        modality = self.modality

        model = self.model

        provider = self.provider

        charged_rates: dict[str, Any] | Unset = UNSET
        if not isinstance(self.charged_rates, Unset):
            charged_rates = self.charged_rates.to_dict()

        charged_tiers: list[dict[str, Any]] | None | Unset
        if isinstance(self.charged_tiers, Unset):
            charged_tiers = UNSET
        elif isinstance(self.charged_tiers, list):
            charged_tiers = []
            for charged_tiers_type_0_item_data in self.charged_tiers:
                charged_tiers_type_0_item = charged_tiers_type_0_item_data.to_dict()
                charged_tiers.append(charged_tiers_type_0_item)

        else:
            charged_tiers = self.charged_tiers

        effective_at: str | Unset = UNSET
        if not isinstance(self.effective_at, Unset):
            effective_at = self.effective_at.isoformat()

        free = self.free

        local = self.local

        markup_bps = self.markup_bps

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

        version = self.version

        when: dict[str, Any] | Unset = UNSET
        if not isinstance(self.when, Unset):
            when = self.when.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "charged_markup_bps": charged_markup_bps,
                "modality": modality,
                "model": model,
                "provider": provider,
            }
        )
        if charged_rates is not UNSET:
            field_dict["charged_rates"] = charged_rates
        if charged_tiers is not UNSET:
            field_dict["charged_tiers"] = charged_tiers
        if effective_at is not UNSET:
            field_dict["effective_at"] = effective_at
        if free is not UNSET:
            field_dict["free"] = free
        if local is not UNSET:
            field_dict["local"] = local
        if markup_bps is not UNSET:
            field_dict["markup_bps"] = markup_bps
        if rates is not UNSET:
            field_dict["rates"] = rates
        if tiers is not UNSET:
            field_dict["tiers"] = tiers
        if version is not UNSET:
            field_dict["version"] = version
        if when is not UNSET:
            field_dict["when"] = when

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_entry_charged_rates import PriceEntryChargedRates
        from ..models.price_entry_rates import PriceEntryRates
        from ..models.price_entry_when import PriceEntryWhen
        from ..models.price_tier import PriceTier

        d = dict(src_dict)
        charged_markup_bps = d.pop("charged_markup_bps")

        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

        _charged_rates = d.pop("charged_rates", UNSET)
        charged_rates: PriceEntryChargedRates | Unset
        if isinstance(_charged_rates, Unset):
            charged_rates = UNSET
        else:
            charged_rates = PriceEntryChargedRates.from_dict(_charged_rates)

        def _parse_charged_tiers(data: object) -> list[PriceTier] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                charged_tiers_type_0 = []
                _charged_tiers_type_0 = data
                for charged_tiers_type_0_item_data in _charged_tiers_type_0:
                    charged_tiers_type_0_item = PriceTier.from_dict(
                        charged_tiers_type_0_item_data
                    )

                    charged_tiers_type_0.append(charged_tiers_type_0_item)

                return charged_tiers_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[PriceTier] | None | Unset, data)

        charged_tiers = _parse_charged_tiers(d.pop("charged_tiers", UNSET))

        _effective_at = d.pop("effective_at", UNSET)
        effective_at: datetime.datetime | Unset
        if isinstance(_effective_at, Unset):
            effective_at = UNSET
        else:
            effective_at = datetime.datetime.fromisoformat(_effective_at)

        free = d.pop("free", UNSET)

        local = d.pop("local", UNSET)

        markup_bps = d.pop("markup_bps", UNSET)

        _rates = d.pop("rates", UNSET)
        rates: PriceEntryRates | Unset
        if isinstance(_rates, Unset):
            rates = UNSET
        else:
            rates = PriceEntryRates.from_dict(_rates)

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
            charged_markup_bps=charged_markup_bps,
            modality=modality,
            model=model,
            provider=provider,
            charged_rates=charged_rates,
            charged_tiers=charged_tiers,
            effective_at=effective_at,
            free=free,
            local=local,
            markup_bps=markup_bps,
            rates=rates,
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
