from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="SpendHardLimitWindow")


@_attrs_define
class SpendHardLimitWindow:
    """
    Attributes:
        limit_micros (int):
        room_micros (int): What it may still spend in the window: the limit less what it spent and what work in flight
            holds.
        spent_micros (int): What the organization spent in the window, as the usage records count it.
        window_end (datetime.datetime):
    """

    limit_micros: int
    room_micros: int
    spent_micros: int
    window_end: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        limit_micros = self.limit_micros

        room_micros = self.room_micros

        spent_micros = self.spent_micros

        window_end = self.window_end.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "limit_micros": limit_micros,
                "room_micros": room_micros,
                "spent_micros": spent_micros,
                "window_end": window_end,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        limit_micros = d.pop("limit_micros")

        room_micros = d.pop("room_micros")

        spent_micros = d.pop("spent_micros")

        window_end = datetime.datetime.fromisoformat(d.pop("window_end"))

        spend_hard_limit_window = cls(
            limit_micros=limit_micros,
            room_micros=room_micros,
            spent_micros=spent_micros,
            window_end=window_end,
        )

        spend_hard_limit_window.additional_properties = d
        return spend_hard_limit_window

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
