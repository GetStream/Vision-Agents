from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.session_sort_direction import SessionSortDirection
from ..models.session_sort_field import SessionSortField
from ..types import UNSET, Unset

T = TypeVar("T", bound="SessionSort")


@_attrs_define
class SessionSort:
    """
    Attributes:
        field (SessionSortField): last_message_at is when the agent last answered, or when the session began if it has
            not. pinned lists the pinned sessions first: it sorts descending only, comes first, and is followed by
            last_message_at, descending when left out. review_score is the review's score, sessions without one last either
            way, and is a server-side caller's only. relevance is the best match first, and is the one sort of a text
            search.
        direction (SessionSortDirection | Unset): -1 is descending and 1 ascending. pinned and relevance only sort
            descending. Default: SessionSortDirection.VALUE_NEGATIVE_1.
    """

    field: SessionSortField
    direction: SessionSortDirection | Unset = SessionSortDirection.VALUE_NEGATIVE_1

    def to_dict(self) -> dict[str, Any]:
        field = self.field.value

        direction: int | Unset = UNSET
        if not isinstance(self.direction, Unset):
            direction = self.direction.value

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "field": field,
            }
        )
        if direction is not UNSET:
            field_dict["direction"] = direction

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        field = SessionSortField._decode(d.pop("field"))

        _direction = d.pop("direction", UNSET)
        direction: SessionSortDirection | Unset
        if isinstance(_direction, Unset):
            direction = UNSET
        else:
            direction = SessionSortDirection(_direction)

        session_sort = cls(
            field=field,
            direction=direction,
        )

        return session_sort
