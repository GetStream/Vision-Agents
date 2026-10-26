from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.price_entry import PriceEntry
    from ..models.price_item import PriceItem
    from ..models.upcoming_price_entry import UpcomingPriceEntry


T = TypeVar("T", bound="PriceList")


@_attrs_define
class PriceList:
    """
    Attributes:
        at (datetime.datetime): The moment the list is read at.
        entries (list[PriceEntry]):
        markup_bps (int): The list's markup in basis points, added to every entry with none of its own.
        unpriced (list[PriceItem]): The items the routing config routes whose work is billable and that no entry prices:
            each records its rows at nothing until a change prices it, or marks it free.
        upcoming (list[UpcomingPriceEntry]): The entries of the changes not cancelled that take effect after at, soonest
            first.
    """

    at: datetime.datetime
    entries: list[PriceEntry]
    markup_bps: int
    unpriced: list[PriceItem]
    upcoming: list[UpcomingPriceEntry]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        at = self.at.isoformat()

        entries = []
        for entries_item_data in self.entries:
            entries_item = entries_item_data.to_dict()
            entries.append(entries_item)

        markup_bps = self.markup_bps

        unpriced = []
        for unpriced_item_data in self.unpriced:
            unpriced_item = unpriced_item_data.to_dict()
            unpriced.append(unpriced_item)

        upcoming = []
        for upcoming_item_data in self.upcoming:
            upcoming_item = upcoming_item_data.to_dict()
            upcoming.append(upcoming_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "at": at,
                "entries": entries,
                "markup_bps": markup_bps,
                "unpriced": unpriced,
                "upcoming": upcoming,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_entry import PriceEntry
        from ..models.price_item import PriceItem
        from ..models.upcoming_price_entry import UpcomingPriceEntry

        d = dict(src_dict)
        at = datetime.datetime.fromisoformat(d.pop("at"))

        entries = []
        _entries = d.pop("entries")
        for entries_item_data in _entries:
            entries_item = PriceEntry.from_dict(entries_item_data)

            entries.append(entries_item)

        markup_bps = d.pop("markup_bps")

        unpriced = []
        _unpriced = d.pop("unpriced")
        for unpriced_item_data in _unpriced:
            unpriced_item = PriceItem.from_dict(unpriced_item_data)

            unpriced.append(unpriced_item)

        upcoming = []
        _upcoming = d.pop("upcoming")
        for upcoming_item_data in _upcoming:
            upcoming_item = UpcomingPriceEntry.from_dict(upcoming_item_data)

            upcoming.append(upcoming_item)

        price_list = cls(
            at=at,
            entries=entries,
            markup_bps=markup_bps,
            unpriced=unpriced,
            upcoming=upcoming,
        )

        price_list.additional_properties = d
        return price_list

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
