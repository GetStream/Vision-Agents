from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.activity_step_day_step import ActivityStepDayStep

T = TypeVar("T", bound="ActivityStepDay")


@_attrs_define
class ActivityStepDay:
    """
    Attributes:
        first_answered (datetime.date): The UTC day, YYYY-MM-DD, of the first request answered in the step's area, over
            every day recorded, not only from since on.
        step (ActivityStepDayStep): The step, one of furthest_step's but none and steady_use.
    """

    first_answered: datetime.date
    step: ActivityStepDayStep
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        first_answered = self.first_answered.isoformat()

        step = self.step.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "first_answered": first_answered,
                "step": step,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        first_answered = datetime.date.fromisoformat(d.pop("first_answered"))

        step = ActivityStepDayStep._decode(d.pop("step"))

        activity_step_day = cls(
            first_answered=first_answered,
            step=step,
        )

        activity_step_day.additional_properties = d
        return activity_step_day

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
