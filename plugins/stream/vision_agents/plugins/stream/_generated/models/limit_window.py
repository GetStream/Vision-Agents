from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.limit_interval import LimitInterval
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.limit_in_force import LimitInForce


T = TypeVar("T", bound="LimitWindow")


@_attrs_define
class LimitWindow:
    """
    Attributes:
        baseline_micros (int): What the subject had spent in the window when it was last reset; its limit counts spend
            past this.
        interval (LimitInterval): The window a limit counts spend over, in UTC: the day, the week from Monday, 14 days
            counted from Monday 2024-01-01, or the calendar month.
        on_grant (bool): The limit is spent and the subject's work is admitted on one of its grants.
        spent_micros (int): What the subject spent in the window, as the usage records count it.
        window_end (datetime.datetime):
        window_start (datetime.datetime):
        limit (LimitInForce | Unset):
        room_micros (int | Unset): What the subject may still spend under its own limit in the window: the limit less
            what it spent since its baseline. Absent with no limit.
    """

    baseline_micros: int
    interval: LimitInterval
    on_grant: bool
    spent_micros: int
    window_end: datetime.datetime
    window_start: datetime.datetime
    limit: LimitInForce | Unset = UNSET
    room_micros: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        baseline_micros = self.baseline_micros

        interval = self.interval.value

        on_grant = self.on_grant

        spent_micros = self.spent_micros

        window_end = self.window_end.isoformat()

        window_start = self.window_start.isoformat()

        limit: dict[str, Any] | Unset = UNSET
        if not isinstance(self.limit, Unset):
            limit = self.limit.to_dict()

        room_micros = self.room_micros

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "baseline_micros": baseline_micros,
                "interval": interval,
                "on_grant": on_grant,
                "spent_micros": spent_micros,
                "window_end": window_end,
                "window_start": window_start,
            }
        )
        if limit is not UNSET:
            field_dict["limit"] = limit
        if room_micros is not UNSET:
            field_dict["room_micros"] = room_micros

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.limit_in_force import LimitInForce

        d = dict(src_dict)
        baseline_micros = d.pop("baseline_micros")

        interval = LimitInterval(d.pop("interval"))

        on_grant = d.pop("on_grant")

        spent_micros = d.pop("spent_micros")

        window_end = datetime.datetime.fromisoformat(d.pop("window_end"))

        window_start = datetime.datetime.fromisoformat(d.pop("window_start"))

        _limit = d.pop("limit", UNSET)
        limit: LimitInForce | Unset
        if isinstance(_limit, Unset):
            limit = UNSET
        else:
            limit = LimitInForce.from_dict(_limit)

        room_micros = d.pop("room_micros", UNSET)

        limit_window = cls(
            baseline_micros=baseline_micros,
            interval=interval,
            on_grant=on_grant,
            spent_micros=spent_micros,
            window_end=window_end,
            window_start=window_start,
            limit=limit,
            room_micros=room_micros,
        )

        limit_window.additional_properties = d
        return limit_window

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
