from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

T = TypeVar("T", bound="TextMatch")


@_attrs_define
class TextMatch:
    """
    Attributes:
        q (str): Quoted phrases and bare words both work, and punctuation is taken rather than refused.
    """

    q: str

    def to_dict(self) -> dict[str, Any]:
        q = self.q

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "$q": q,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        q = d.pop("$q")

        text_match = cls(
            q=q,
        )

        return text_match
