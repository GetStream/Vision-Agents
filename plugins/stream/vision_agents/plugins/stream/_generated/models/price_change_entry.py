from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
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
        free (bool | Unset):
        markup_bps (int | Unset): The entry's own markup in basis points. Absent takes the list's.
        rates (PriceChangeEntryRates | Unset): The vendor's rates, by meter.
        tiers (list[PriceTier] | None | Unset):
        when (PriceChangeEntryWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}.
            Absent is the item's default entry.
        withdrawn (bool | Unset): The change stopped pricing the item under its when.
    """

    modality: str
    model: str
    provider: str
    free: bool | Unset = UNSET
    markup_bps: int | Unset = UNSET
    rates: PriceChangeEntryRates | Unset = UNSET
    tiers: list[PriceTier] | None | Unset = UNSET
    when: PriceChangeEntryWhen | Unset = UNSET
    withdrawn: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        modality = self.modality

        model = self.model

        provider = self.provider

        free = self.free

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
            }
        )
        if free is not UNSET:
            field_dict["free"] = free
        if markup_bps is not UNSET:
            field_dict["markup_bps"] = markup_bps
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
        from ..models.price_change_entry_rates import PriceChangeEntryRates
        from ..models.price_change_entry_when import PriceChangeEntryWhen
        from ..models.price_tier import PriceTier

        d = dict(src_dict)
        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

        free = d.pop("free", UNSET)

        markup_bps = d.pop("markup_bps", UNSET)

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
            free=free,
            markup_bps=markup_bps,
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
