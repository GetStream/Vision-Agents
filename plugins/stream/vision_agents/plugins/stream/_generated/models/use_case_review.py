from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.review_actor import ReviewActor
from ..models.use_case_status import UseCaseStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.use_case_review_vendor_payload import UseCaseReviewVendorPayload


T = TypeVar("T", bound="UseCaseReview")


@_attrs_define
class UseCaseReview:
    """One move a use case made, and who made it. Nothing here is ever changed.

    Attributes:
        actor (ReviewActor): Who moved a use case: the app, Stream staff or the vendor.
        created_at (datetime.datetime):
        from_status (UseCaseStatus): Where a use case stands. draft, changes_requested and vendor_rejected can be edited
            and submitted; submitted waits on Stream's review; vendor_pending on the vendor's; approved numbers may send.
            rejected is final.
        id (str):
        to_status (UseCaseStatus): Where a use case stands. draft, changes_requested and vendor_rejected can be edited
            and submitted; submitted waits on Stream's review; vendor_pending on the vendor's; approved numbers may send.
            rejected is final.
        actor_name (str | Unset): Who exactly: the reviewer, or the vendor.
        approved_at (datetime.datetime | Unset):
        notes (str | Unset): What the reviewer or vendor said.
        submitted_at (datetime.datetime | Unset):
        vendor_payload (UseCaseReviewVendorPayload | Unset): What the vendor answered, as it came.
    """

    actor: ReviewActor
    created_at: datetime.datetime
    from_status: UseCaseStatus
    id: str
    to_status: UseCaseStatus
    actor_name: str | Unset = UNSET
    approved_at: datetime.datetime | Unset = UNSET
    notes: str | Unset = UNSET
    submitted_at: datetime.datetime | Unset = UNSET
    vendor_payload: UseCaseReviewVendorPayload | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        actor = self.actor.value

        created_at = self.created_at.isoformat()

        from_status = self.from_status.value

        id = self.id

        to_status = self.to_status.value

        actor_name = self.actor_name

        approved_at: str | Unset = UNSET
        if not isinstance(self.approved_at, Unset):
            approved_at = self.approved_at.isoformat()

        notes = self.notes

        submitted_at: str | Unset = UNSET
        if not isinstance(self.submitted_at, Unset):
            submitted_at = self.submitted_at.isoformat()

        vendor_payload: dict[str, Any] | Unset = UNSET
        if not isinstance(self.vendor_payload, Unset):
            vendor_payload = self.vendor_payload.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "actor": actor,
                "created_at": created_at,
                "from_status": from_status,
                "id": id,
                "to_status": to_status,
            }
        )
        if actor_name is not UNSET:
            field_dict["actor_name"] = actor_name
        if approved_at is not UNSET:
            field_dict["approved_at"] = approved_at
        if notes is not UNSET:
            field_dict["notes"] = notes
        if submitted_at is not UNSET:
            field_dict["submitted_at"] = submitted_at
        if vendor_payload is not UNSET:
            field_dict["vendor_payload"] = vendor_payload

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.use_case_review_vendor_payload import (
            UseCaseReviewVendorPayload,
        )

        d = dict(src_dict)
        actor = ReviewActor(d.pop("actor"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        from_status = UseCaseStatus(d.pop("from_status"))

        id = d.pop("id")

        to_status = UseCaseStatus(d.pop("to_status"))

        actor_name = d.pop("actor_name", UNSET)

        _approved_at = d.pop("approved_at", UNSET)
        approved_at: datetime.datetime | Unset
        if isinstance(_approved_at, Unset):
            approved_at = UNSET
        else:
            approved_at = datetime.datetime.fromisoformat(_approved_at)

        notes = d.pop("notes", UNSET)

        _submitted_at = d.pop("submitted_at", UNSET)
        submitted_at: datetime.datetime | Unset
        if isinstance(_submitted_at, Unset):
            submitted_at = UNSET
        else:
            submitted_at = datetime.datetime.fromisoformat(_submitted_at)

        _vendor_payload = d.pop("vendor_payload", UNSET)
        vendor_payload: UseCaseReviewVendorPayload | Unset
        if isinstance(_vendor_payload, Unset):
            vendor_payload = UNSET
        else:
            vendor_payload = UseCaseReviewVendorPayload.from_dict(_vendor_payload)

        use_case_review = cls(
            actor=actor,
            created_at=created_at,
            from_status=from_status,
            id=id,
            to_status=to_status,
            actor_name=actor_name,
            approved_at=approved_at,
            notes=notes,
            submitted_at=submitted_at,
            vendor_payload=vendor_payload,
        )

        use_case_review.additional_properties = d
        return use_case_review

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
