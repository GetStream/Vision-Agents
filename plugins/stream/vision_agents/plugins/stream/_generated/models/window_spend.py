from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.limit_interval import LimitInterval

T = TypeVar("T", bound="WindowSpend")


@_attrs_define
class WindowSpend:
    """
    Attributes:
        interval (LimitInterval): The window a limit counts spend over, in UTC: the day, the week from Monday, 14 days
            counted from Monday 2024-01-01, or the calendar month.
        spent_micros (int):
        window_end (datetime.datetime):
        window_start (datetime.datetime):
    """

    interval: LimitInterval
    spent_micros: int
    window_end: datetime.datetime
    window_start: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        interval = self.interval.value

        spent_micros = self.spent_micros

        window_end = self.window_end.isoformat()

        window_start = self.window_start.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "interval": interval,
                "spent_micros": spent_micros,
                "window_end": window_end,
                "window_start": window_start,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        interval = LimitInterval(d.pop("interval"))

        spent_micros = d.pop("spent_micros")

        window_end = datetime.datetime.fromisoformat(d.pop("window_end"))

        window_start = datetime.datetime.fromisoformat(d.pop("window_start"))

        window_spend = cls(
            interval=interval,
            spent_micros=spent_micros,
            window_end=window_end,
            window_start=window_start,
        )

        window_spend.additional_properties = d
        return window_spend

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
