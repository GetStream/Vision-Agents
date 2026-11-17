from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.review_cause import ReviewCause


T = TypeVar("T", bound="ReviewAnalysis")


@_attrs_define
class ReviewAnalysis:
    """
    Attributes:
        calls_read (int): How many badly reviewed calls were read. Zero means there were none, and nothing was asked.
        causes (list[ReviewCause]):
        summary (str): The whole, in two sentences.
    """

    calls_read: int
    causes: list[ReviewCause]
    summary: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        calls_read = self.calls_read

        causes = []
        for causes_item_data in self.causes:
            causes_item = causes_item_data.to_dict()
            causes.append(causes_item)

        summary = self.summary

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "calls_read": calls_read,
                "causes": causes,
                "summary": summary,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.review_cause import ReviewCause

        d = dict(src_dict)
        calls_read = d.pop("calls_read")

        causes = []
        _causes = d.pop("causes")
        for causes_item_data in _causes:
            causes_item = ReviewCause.from_dict(causes_item_data)

            causes.append(causes_item)

        summary = d.pop("summary")

        review_analysis = cls(
            calls_read=calls_read,
            causes=causes,
            summary=summary,
        )

        review_analysis.additional_properties = d
        return review_analysis

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
