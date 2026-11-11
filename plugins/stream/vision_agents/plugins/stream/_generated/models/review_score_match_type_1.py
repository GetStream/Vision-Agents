from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ReviewScoreMatchType1")


@_attrs_define
class ReviewScoreMatchType1:
    """
    Attributes:
        eq (int | Unset):
        gt (int | Unset):
        gte (int | Unset):
        lt (int | Unset):
        lte (int | Unset):
    """

    eq: int | Unset = UNSET
    gt: int | Unset = UNSET
    gte: int | Unset = UNSET
    lt: int | Unset = UNSET
    lte: int | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        eq = self.eq

        gt = self.gt

        gte = self.gte

        lt = self.lt

        lte = self.lte

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if eq is not UNSET:
            field_dict["$eq"] = eq
        if gt is not UNSET:
            field_dict["$gt"] = gt
        if gte is not UNSET:
            field_dict["$gte"] = gte
        if lt is not UNSET:
            field_dict["$lt"] = lt
        if lte is not UNSET:
            field_dict["$lte"] = lte

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        eq = d.pop("$eq", UNSET)

        gt = d.pop("$gt", UNSET)

        gte = d.pop("$gte", UNSET)

        lt = d.pop("$lt", UNSET)

        lte = d.pop("$lte", UNSET)

        review_score_match_type_1 = cls(
            eq=eq,
            gt=gt,
            gte=gte,
            lt=lt,
            lte=lte,
        )

        return review_score_match_type_1
