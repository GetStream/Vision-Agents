from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="LimitChange")


@_attrs_define
class LimitChange:
    """What changes about a spend limit: its amount and its note. A field left out keeps what is stored.

    Attributes:
        limit_micros (int | Unset):
        note (str | Unset):
    """

    limit_micros: int | Unset = UNSET
    note: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        limit_micros = self.limit_micros

        note = self.note

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if limit_micros is not UNSET:
            field_dict["limit_micros"] = limit_micros
        if note is not UNSET:
            field_dict["note"] = note

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        limit_micros = d.pop("limit_micros", UNSET)

        note = d.pop("note", UNSET)

        limit_change = cls(
            limit_micros=limit_micros,
            note=note,
        )

        return limit_change
