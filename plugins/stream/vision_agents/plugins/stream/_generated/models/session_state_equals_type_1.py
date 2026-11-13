from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.session_state import SessionState

T = TypeVar("T", bound="SessionStateEqualsType1")


@_attrs_define
class SessionStateEqualsType1:
    """
    Attributes:
        eq (SessionState): Whether the agent is still in the call.
    """

    eq: SessionState

    def to_dict(self) -> dict[str, Any]:
        eq = self.eq.value

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
        eq = SessionState._decode(d.pop("$eq"))

        session_state_equals_type_1 = cls(
            eq=eq,
        )

        return session_state_equals_type_1
