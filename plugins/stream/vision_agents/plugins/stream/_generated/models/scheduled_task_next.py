from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ScheduledTaskNext")


@_attrs_define
class ScheduledTaskNext:
    """
    Attributes:
        at (str | Unset): The next occurrence as the user's clock reads it, in the task's zone.
        in_ (str | Unset): How long from now instead, an ISO 8601 duration: PT15M. Example: PT15M.
    """

    at: str | Unset = UNSET
    in_: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        at = self.at

        in_ = self.in_

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if at is not UNSET:
            field_dict["at"] = at
        if in_ is not UNSET:
            field_dict["in"] = in_

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        at = d.pop("at", UNSET)

        in_ = d.pop("in", UNSET)

        scheduled_task_next = cls(
            at=at,
            in_=in_,
        )

        scheduled_task_next.additional_properties = d
        return scheduled_task_next

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
