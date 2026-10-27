from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_entry_input import PriceEntryInput


T = TypeVar("T", bound="PriceChangeInput")


@_attrs_define
class PriceChangeInput:
    """
    Attributes:
        name (str): 1 to 48 characters of a-z, 0-9 and _. The change is named with the UTC time it was made before it.
        reason (str): What changed, from what to what. At least one sentence; trimmed, and refused when blank.
        source (str): The vendor's page and the date it was read, or the operator's own decision. Trimmed, and refused
            when blank.
        confirm (bool | Unset): Required when the preview answers confirmation_required.
        effective_at (datetime.datetime | Unset): When the change takes effect. Absent is now; it may be up to two years
            ahead, but never more than 5 minutes in the past, and a time up to 5 minutes past is taken as now.
        entries (list[PriceEntryInput] | None | Unset): Each replaces its item's entry under its when whole: its cost,
            its price and every tier of both.
    """

    name: str
    reason: str
    source: str
    confirm: bool | Unset = UNSET
    effective_at: datetime.datetime | Unset = UNSET
    entries: list[PriceEntryInput] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        reason = self.reason

        source = self.source

        confirm = self.confirm

        effective_at: str | Unset = UNSET
        if not isinstance(self.effective_at, Unset):
            effective_at = self.effective_at.isoformat()

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

        field_dict.update(
            {
                "name": name,
                "reason": reason,
                "source": source,
            }
        )
        if confirm is not UNSET:
            field_dict["confirm"] = confirm
        if effective_at is not UNSET:
            field_dict["effective_at"] = effective_at
        if entries is not UNSET:
            field_dict["entries"] = entries

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_entry_input import PriceEntryInput

        d = dict(src_dict)
        name = d.pop("name")

        reason = d.pop("reason")

        source = d.pop("source")

        confirm = d.pop("confirm", UNSET)

        _effective_at = d.pop("effective_at", UNSET)
        effective_at: datetime.datetime | Unset
        if isinstance(_effective_at, Unset):
            effective_at = UNSET
        else:
            effective_at = datetime.datetime.fromisoformat(_effective_at)

        def _parse_entries(data: object) -> list[PriceEntryInput] | None | Unset:
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
                    entries_type_0_item = PriceEntryInput.from_dict(
                        entries_type_0_item_data
                    )

                    entries_type_0.append(entries_type_0_item)

                return entries_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[PriceEntryInput] | None | Unset, data)

        entries = _parse_entries(d.pop("entries", UNSET))

        price_change_input = cls(
            name=name,
            reason=reason,
            source=source,
            confirm=confirm,
            effective_at=effective_at,
            entries=entries,
        )

        return price_change_input
