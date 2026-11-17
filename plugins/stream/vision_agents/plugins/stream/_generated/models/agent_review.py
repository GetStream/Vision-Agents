from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentReview")


@_attrs_define
class AgentReview:
    """How a finished call under an agent is reviewed.

    Attributes:
        instructions (str | Unset): What the model reviewing a finished call judges it by, in place of Stream's default
            guide: what to summarise and how to score the call from 1 to 5. A score outside 1 to 5 is not kept. The reviewer
            is always told to answer with JSON holding a summary, a score and notes, so leave the answer's format out. Empty
            is the default guide. On an update or a patch, left out keeps the stored instructions and an empty string
            removes them.
    """

    instructions: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        instructions = self.instructions

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if instructions is not UNSET:
            field_dict["instructions"] = instructions

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        instructions = d.pop("instructions", UNSET)

        agent_review = cls(
            instructions=instructions,
        )

        agent_review.additional_properties = d
        return agent_review

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
