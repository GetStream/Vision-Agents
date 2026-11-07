from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ContactCounts")


@_attrs_define
class ContactCounts:
    """
    Attributes:
        completed (int):
        converted (int): Contacts that ended successfully.
        dialed (int): Contacts with at least one call placed.
        enrolled (int):
        failed (int):
        in_progress (int):
        rejected (int):
        stopped (int):
        total (int):
    """

    completed: int
    converted: int
    dialed: int
    enrolled: int
    failed: int
    in_progress: int
    rejected: int
    stopped: int
    total: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        completed = self.completed

        converted = self.converted

        dialed = self.dialed

        enrolled = self.enrolled

        failed = self.failed

        in_progress = self.in_progress

        rejected = self.rejected

        stopped = self.stopped

        total = self.total

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "completed": completed,
                "converted": converted,
                "dialed": dialed,
                "enrolled": enrolled,
                "failed": failed,
                "in_progress": in_progress,
                "rejected": rejected,
                "stopped": stopped,
                "total": total,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        completed = d.pop("completed")

        converted = d.pop("converted")

        dialed = d.pop("dialed")

        enrolled = d.pop("enrolled")

        failed = d.pop("failed")

        in_progress = d.pop("in_progress")

        rejected = d.pop("rejected")

        stopped = d.pop("stopped")

        total = d.pop("total")

        contact_counts = cls(
            completed=completed,
            converted=converted,
            dialed=dialed,
            enrolled=enrolled,
            failed=failed,
            in_progress=in_progress,
            rejected=rejected,
            stopped=stopped,
            total=total,
        )

        contact_counts.additional_properties = d
        return contact_counts

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
