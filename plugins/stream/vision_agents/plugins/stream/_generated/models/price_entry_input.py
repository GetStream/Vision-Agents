from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
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
        free (bool | Unset): The vendor charges nothing for it.
        local (bool | Unset): Refused with invalid_price: a model is local when the routing config says so.
        markup_bps (int | Unset): The entry's own markup in basis points, 0 to 100000, replacing the list's. Absent
            takes the list's.
        rates (PriceEntryInputRates | Unset): The vendor's rates, by meter: input_tokens, cached_input_tokens,
            cache_write_input_tokens, output_tokens, characters, audio_ms, images, requests, message_segments, call_seconds
            and the rest pricing knows. An unknown meter, and a rate of more than $1000 a unit, are refused.
        tiers (list[PriceTier] | None | Unset): Rates for long prompts. Each tier rates every meter the entry does.
        version (str | Unset): Refused with invalid_price: an entry's version is the change that writes it.
        when (PriceEntryInputWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}.
            Absent is the item's default entry.
        withdrawn (bool | Unset): Stop pricing the item under its when from effective_at on. Its entries under another
            when are untouched. A withdrawal carries no rate.
    """

    modality: str
    model: str
    provider: str
    free: bool | Unset = UNSET
    local: bool | Unset = UNSET
    markup_bps: int | Unset = UNSET
    rates: PriceEntryInputRates | Unset = UNSET
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
        if withdrawn is not UNSET:
            field_dict["withdrawn"] = withdrawn

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_entry_input_rates import PriceEntryInputRates
        from ..models.price_entry_input_when import PriceEntryInputWhen
        from ..models.price_tier import PriceTier

        d = dict(src_dict)
        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

        free = d.pop("free", UNSET)

        local = d.pop("local", UNSET)

        markup_bps = d.pop("markup_bps", UNSET)

        _rates = d.pop("rates", UNSET)
        rates: PriceEntryInputRates | Unset
        if isinstance(_rates, Unset):
            rates = UNSET
        else:
            rates = PriceEntryInputRates.from_dict(_rates)

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
            markup_bps=markup_bps,
            rates=rates,
            tiers=tiers,
            version=version,
            when=when,
            withdrawn=withdrawn,
        )

        return price_entry_input
