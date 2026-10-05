from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.budget_interval import BudgetInterval
from ..types import UNSET, Unset

T = TypeVar("T", bound="Budget")


@_attrs_define
class Budget:
    """A cap on spend across every modality, reset on a UTC boundary each interval. Once it is spent every new session and
    every LLM response is refused until the next interval. Checks are cached for a few seconds, so a busy app can
    overshoot by what it spends in that time.

        Attributes:
            interval (BudgetInterval): How often a budget resets. A week starts on Monday.
            limit_micros (int): The cap, in millionths of a dollar. Example: 100000000.
            resets_at (datetime.datetime | Unset): When the current interval ends.
            spent_micros (int | Unset): What has been spent in the current interval.
    """

    interval: BudgetInterval
    limit_micros: int
    resets_at: datetime.datetime | Unset = UNSET
    spent_micros: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        interval = self.interval.value

        limit_micros = self.limit_micros

        resets_at: str | Unset = UNSET
        if not isinstance(self.resets_at, Unset):
            resets_at = self.resets_at.isoformat()

        spent_micros = self.spent_micros

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "interval": interval,
                "limit_micros": limit_micros,
            }
        )
        if resets_at is not UNSET:
            field_dict["resets_at"] = resets_at
        if spent_micros is not UNSET:
            field_dict["spent_micros"] = spent_micros

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        interval = BudgetInterval(d.pop("interval"))

        limit_micros = d.pop("limit_micros")

        _resets_at = d.pop("resets_at", UNSET)
        resets_at: datetime.datetime | Unset
        if isinstance(_resets_at, Unset):
            resets_at = UNSET
        else:
            resets_at = datetime.datetime.fromisoformat(_resets_at)

        spent_micros = d.pop("spent_micros", UNSET)

        budget = cls(
            interval=interval,
            limit_micros=limit_micros,
            resets_at=resets_at,
            spent_micros=spent_micros,
        )

        budget.additional_properties = d
        return budget

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
