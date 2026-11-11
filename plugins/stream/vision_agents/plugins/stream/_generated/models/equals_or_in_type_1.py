from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="EqualsOrInType1")


@_attrs_define
class EqualsOrInType1:
    """
    Attributes:
        eq (str | Unset):
        in_ (list[str] | Unset):
    """

    eq: str | Unset = UNSET
    in_: list[str] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        eq = self.eq

        in_: list[str] | Unset = UNSET
        if not isinstance(self.in_, Unset):
            in_ = self.in_

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if eq is not UNSET:
            field_dict["$eq"] = eq
        if in_ is not UNSET:
            field_dict["$in"] = in_

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        eq = d.pop("$eq", UNSET)

        in_ = cast(list[str], d.pop("$in", UNSET))

        equals_or_in_type_1 = cls(
            eq=eq,
            in_=in_,
        )

        return equals_or_in_type_1
