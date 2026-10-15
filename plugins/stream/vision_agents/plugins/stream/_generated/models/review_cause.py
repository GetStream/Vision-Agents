from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ReviewCause")


@_attrs_define
class ReviewCause:
    """
    Attributes:
        call_ids (list[str]): The calls it explains, which are the sessions that ran them. Only ids that were read.
        explanation (str): What went wrong in those calls.
        fix (str): What to change in the agent.
        title (str):
    """

    call_ids: list[str]
    explanation: str
    fix: str
    title: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        call_ids = self.call_ids

        explanation = self.explanation

        fix = self.fix

        title = self.title

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "call_ids": call_ids,
                "explanation": explanation,
                "fix": fix,
                "title": title,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        call_ids = cast(list[str], d.pop("call_ids"))

        explanation = d.pop("explanation")

        fix = d.pop("fix")

        title = d.pop("title")

        review_cause = cls(
            call_ids=call_ids,
            explanation=explanation,
            fix=fix,
            title=title,
        )

        review_cause.additional_properties = d
        return review_cause

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
