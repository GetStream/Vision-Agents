from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.price_change_kind import PriceChangeKind
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_change_entry import PriceChangeEntry


T = TypeVar("T", bound="PriceChange")


@_attrs_define
class PriceChange:
    """
    Attributes:
        created_at (datetime.datetime):
        created_by (str): The actor that made it, or the router release that loaded it.
        effective_at (datetime.datetime):
        kind (PriceChangeKind): What wrote the change: seed is a database's first load from the router's price files,
            release the items a release's price files had that the database lacked, and api a change made through the
            management API.
        name (str): <UTC yyyymmddhhmmss>_<the name it was given>: the price_version of every row its entries price.
        reason (str):
        source (str):
        cancel_reason (str | Unset):
        cancelled_at (datetime.datetime | Unset):
        cancelled_by (str | Unset):
        entries (list[PriceChangeEntry] | None | Unset): The entries it wrote. Absent from a list of changes.
    """

    created_at: datetime.datetime
    created_by: str
    effective_at: datetime.datetime
    kind: PriceChangeKind
    name: str
    reason: str
    source: str
    cancel_reason: str | Unset = UNSET
    cancelled_at: datetime.datetime | Unset = UNSET
    cancelled_by: str | Unset = UNSET
    entries: list[PriceChangeEntry] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        created_by = self.created_by

        effective_at = self.effective_at.isoformat()

        kind = self.kind.value

        name = self.name

        reason = self.reason

        source = self.source

        cancel_reason = self.cancel_reason

        cancelled_at: str | Unset = UNSET
        if not isinstance(self.cancelled_at, Unset):
            cancelled_at = self.cancelled_at.isoformat()

        cancelled_by = self.cancelled_by

        entries: list[dict[str, Any]] | None | Unset
        if isinstance(self.entries, Unset):
            entries = UNSET
        elif isinstance(self.entries, list):
            entries = []
            for entries_type_0_item_data in self.entries:
                entries_type_0_item = entries_type_0_item_data.to_dict()
                entries.append(entries_type_0_item)

        else:
            entries = self.entries

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "created_by": created_by,
                "effective_at": effective_at,
                "kind": kind,
                "name": name,
                "reason": reason,
                "source": source,
            }
        )
        if cancel_reason is not UNSET:
            field_dict["cancel_reason"] = cancel_reason
        if cancelled_at is not UNSET:
            field_dict["cancelled_at"] = cancelled_at
        if cancelled_by is not UNSET:
            field_dict["cancelled_by"] = cancelled_by
        if entries is not UNSET:
            field_dict["entries"] = entries

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_change_entry import PriceChangeEntry

        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        created_by = d.pop("created_by")

        effective_at = datetime.datetime.fromisoformat(d.pop("effective_at"))

        kind = PriceChangeKind(d.pop("kind"))

        name = d.pop("name")

        reason = d.pop("reason")

        source = d.pop("source")

        cancel_reason = d.pop("cancel_reason", UNSET)

        _cancelled_at = d.pop("cancelled_at", UNSET)
        cancelled_at: datetime.datetime | Unset
        if isinstance(_cancelled_at, Unset):
            cancelled_at = UNSET
        else:
            cancelled_at = datetime.datetime.fromisoformat(_cancelled_at)

        cancelled_by = d.pop("cancelled_by", UNSET)

        def _parse_entries(data: object) -> list[PriceChangeEntry] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                entries_type_0 = []
                _entries_type_0 = data
                for entries_type_0_item_data in _entries_type_0:
                    entries_type_0_item = PriceChangeEntry.from_dict(
                        entries_type_0_item_data
                    )

                    entries_type_0.append(entries_type_0_item)

                return entries_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[PriceChangeEntry] | None | Unset, data)

        entries = _parse_entries(d.pop("entries", UNSET))

        price_change = cls(
            created_at=created_at,
            created_by=created_by,
            effective_at=effective_at,
            kind=kind,
            name=name,
            reason=reason,
            source=source,
            cancel_reason=cancel_reason,
            cancelled_at=cancelled_at,
            cancelled_by=cancelled_by,
            entries=entries,
        )

        price_change.additional_properties = d
        return price_change

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
