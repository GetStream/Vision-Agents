from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

T = TypeVar("T", bound="LimitGrantUnrevocation")


@_attrs_define
class LimitGrantUnrevocation:
    """Why a revoked grant is live again.

    Attributes:
        reason (str): Why it is live again; more than spaces.
    """

    reason: str

    def to_dict(self) -> dict[str, Any]:
        reason = self.reason

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "reason": reason,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        reason = d.pop("reason")

        limit_grant_unrevocation = cls(
            reason=reason,
        )

        return limit_grant_unrevocation
