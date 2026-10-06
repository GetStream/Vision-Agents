from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.review_decision import ReviewDecision
from ..types import UNSET, Unset

T = TypeVar("T", bound="ReviewUseCaseRequest")


@_attrs_define
class ReviewUseCaseRequest:
    """
    Attributes:
        decision (ReviewDecision): approve sends the use case to the vendor; request_changes hands it back to the app to
            edit; reject ends it.
        notes (str | Unset): What the app should change, or why it was rejected. Required unless approving.
        reviewer (str | Unset): Who decided, as the app's timeline shows it.
    """

    decision: ReviewDecision
    notes: str | Unset = UNSET
    reviewer: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        decision = self.decision.value

        notes = self.notes

        reviewer = self.reviewer

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "decision": decision,
            }
        )
        if notes is not UNSET:
            field_dict["notes"] = notes
        if reviewer is not UNSET:
            field_dict["reviewer"] = reviewer

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        decision = ReviewDecision(d.pop("decision"))

        notes = d.pop("notes", UNSET)

        reviewer = d.pop("reviewer", UNSET)

        review_use_case_request = cls(
            decision=decision,
            notes=notes,
            reviewer=reviewer,
        )

        review_use_case_request.additional_properties = d
        return review_use_case_request

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
