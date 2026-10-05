from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

T = TypeVar("T", bound="EqualsType1")


@_attrs_define
class EqualsType1:
    """
    Attributes:
        eq (str):
    """

    eq: str

    def to_dict(self) -> dict[str, Any]:
        eq = self.eq

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "$eq": eq,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        eq = d.pop("$eq")

        equals_type_1 = cls(
            eq=eq,
        )

        return equals_type_1
