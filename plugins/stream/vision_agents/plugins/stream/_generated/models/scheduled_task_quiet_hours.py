from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ScheduledTaskQuietHours")


@_attrs_define
class ScheduledTaskQuietHours:
    """
    Attributes:
        end (str): When the hold ends. It may read before the start: 22:00 to 07:00 holds the night. Example: 07:00.
        start (str): When the hold begins, as the user's clock reads it. Both boundaries empty is no hold, which is also
            how an update clears one. Example: 22:00.
    """

    end: str
    start: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        end = self.end

        start = self.start

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "end": end,
                "start": start,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        end = d.pop("end")

        start = d.pop("start")

        scheduled_task_quiet_hours = cls(
            end=end,
            start=start,
        )

        scheduled_task_quiet_hours.additional_properties = d
        return scheduled_task_quiet_hours

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
