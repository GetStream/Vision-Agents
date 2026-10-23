from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_tier import PriceTier
    from ..models.priced_entry_charged_rates import PricedEntryChargedRates
    from ..models.priced_entry_rates import PricedEntryRates


T = TypeVar("T", bound="PricedEntry")


@_attrs_define
class PricedEntry:
    """
    Attributes:
        markup_bps (int): The markup it is sold at.
        charged_rates (PricedEntryChargedRates | Unset):
        charged_tiers (list[PriceTier] | None | Unset):
        free (bool | Unset):
        local (bool | Unset):
        rates (PricedEntryRates | Unset): The vendor's rates.
        tiers (list[PriceTier] | None | Unset):
        version (str | Unset): The change that wrote it. Absent for the change previewed.
    """

    markup_bps: int
    charged_rates: PricedEntryChargedRates | Unset = UNSET
    charged_tiers: list[PriceTier] | None | Unset = UNSET
    free: bool | Unset = UNSET
    local: bool | Unset = UNSET
    rates: PricedEntryRates | Unset = UNSET
    tiers: list[PriceTier] | None | Unset = UNSET
    version: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        markup_bps = self.markup_bps

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

        free = self.free

        local = self.local

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

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "markup_bps": markup_bps,
            }
        )
        if charged_rates is not UNSET:
            field_dict["charged_rates"] = charged_rates
        if charged_tiers is not UNSET:
            field_dict["charged_tiers"] = charged_tiers
        if free is not UNSET:
            field_dict["free"] = free
        if local is not UNSET:
            field_dict["local"] = local
        if rates is not UNSET:
            field_dict["rates"] = rates
        if tiers is not UNSET:
            field_dict["tiers"] = tiers
        if version is not UNSET:
            field_dict["version"] = version

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_tier import PriceTier
        from ..models.priced_entry_charged_rates import PricedEntryChargedRates
        from ..models.priced_entry_rates import PricedEntryRates

        d = dict(src_dict)
        markup_bps = d.pop("markup_bps")

        _charged_rates = d.pop("charged_rates", UNSET)
        charged_rates: PricedEntryChargedRates | Unset
        if isinstance(_charged_rates, Unset):
            charged_rates = UNSET
        else:
            charged_rates = PricedEntryChargedRates.from_dict(_charged_rates)

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

        free = d.pop("free", UNSET)

        local = d.pop("local", UNSET)

        _rates = d.pop("rates", UNSET)
        rates: PricedEntryRates | Unset
        if isinstance(_rates, Unset):
            rates = UNSET
        else:
            rates = PricedEntryRates.from_dict(_rates)

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

        priced_entry = cls(
            markup_bps=markup_bps,
            charged_rates=charged_rates,
            charged_tiers=charged_tiers,
            free=free,
            local=local,
            rates=rates,
            tiers=tiers,
            version=version,
        )

        priced_entry.additional_properties = d
        return priced_entry

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
