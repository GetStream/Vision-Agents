from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="TimeRange")


@_attrs_define
class TimeRange:
    """
    Attributes:
        gte (datetime.datetime | Unset): At or after this RFC3339 time.
        lt (datetime.datetime | Unset): Strictly before this RFC3339 time.
    """

    gte: datetime.datetime | Unset = UNSET
    lt: datetime.datetime | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        gte: str | Unset = UNSET
        if not isinstance(self.gte, Unset):
            gte = self.gte.isoformat()

        lt: str | Unset = UNSET
        if not isinstance(self.lt, Unset):
            lt = self.lt.isoformat()

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if gte is not UNSET:
            field_dict["$gte"] = gte
        if lt is not UNSET:
            field_dict["$lt"] = lt

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        _gte = d.pop("$gte", UNSET)
        gte: datetime.datetime | Unset
        if isinstance(_gte, Unset):
            gte = UNSET
        else:
            gte = datetime.datetime.fromisoformat(_gte)

        _lt = d.pop("$lt", UNSET)
        lt: datetime.datetime | Unset
        if isinstance(_lt, Unset):
            lt = UNSET
        else:
            lt = datetime.datetime.fromisoformat(_lt)

        time_range = cls(
            gte=gte,
            lt=lt,
        )

        return time_range
