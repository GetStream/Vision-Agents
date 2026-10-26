from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.price import Price


T = TypeVar("T", bound="Prices")


@_attrs_define
class Prices:
    """
    Attributes:
        at (datetime.datetime): The moment the list is read at.
        entries (list[Price]): Every item priced now, by modality, provider and model.
        upcoming (list[Price]): What the changes scheduled after at will set, soonest first.
    """

    at: datetime.datetime
    entries: list[Price]
    upcoming: list[Price]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        at = self.at.isoformat()

        entries = []
        for entries_item_data in self.entries:
            entries_item = entries_item_data.to_dict()
            entries.append(entries_item)

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
                "upcoming": upcoming,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price import Price

        d = dict(src_dict)
        at = datetime.datetime.fromisoformat(d.pop("at"))

        entries = []
        _entries = d.pop("entries")
        for entries_item_data in _entries:
            entries_item = Price.from_dict(entries_item_data)

            entries.append(entries_item)

        upcoming = []
        _upcoming = d.pop("upcoming")
        for upcoming_item_data in _upcoming:
            upcoming_item = Price.from_dict(upcoming_item_data)

            upcoming.append(upcoming_item)

        prices = cls(
            at=at,
            entries=entries,
            upcoming=upcoming,
        )

        prices.additional_properties = d
        return prices

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
