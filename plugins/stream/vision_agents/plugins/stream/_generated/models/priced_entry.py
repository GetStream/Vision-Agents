from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_tier import PriceTier
    from ..models.priced_entry_prices import PricedEntryPrices
    from ..models.priced_entry_rates import PricedEntryRates


T = TypeVar("T", bound="PricedEntry")


@_attrs_define
class PricedEntry:
    """
    Attributes:
        free (bool | Unset):
        local (bool | Unset):
        price_tiers (list[PriceTier] | None | Unset):
        prices (PricedEntryPrices | Unset): What customers are charged.
        rates (PricedEntryRates | Unset): The vendor's cost, for reference.
        sku (str | Unset): The SKU it is priced under.
        tiers (list[PriceTier] | None | Unset):
        version (str | Unset): The change that wrote it. Absent for the change previewed.
    """

    free: bool | Unset = UNSET
    local: bool | Unset = UNSET
    price_tiers: list[PriceTier] | None | Unset = UNSET
    prices: PricedEntryPrices | Unset = UNSET
    rates: PricedEntryRates | Unset = UNSET
    sku: str | Unset = UNSET
    tiers: list[PriceTier] | None | Unset = UNSET
    version: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
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

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
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

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_tier import PriceTier
        from ..models.priced_entry_prices import PricedEntryPrices
        from ..models.priced_entry_rates import PricedEntryRates

        d = dict(src_dict)
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
        prices: PricedEntryPrices | Unset
        if isinstance(_prices, Unset):
            prices = UNSET
        else:
            prices = PricedEntryPrices.from_dict(_prices)

        _rates = d.pop("rates", UNSET)
        rates: PricedEntryRates | Unset
        if isinstance(_rates, Unset):
            rates = UNSET
        else:
            rates = PricedEntryRates.from_dict(_rates)

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

        priced_entry = cls(
            free=free,
            local=local,
            price_tiers=price_tiers,
            prices=prices,
            rates=rates,
            sku=sku,
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
