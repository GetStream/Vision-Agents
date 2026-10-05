from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.granularity import Granularity

T = TypeVar("T", bound="RollupResult")


@_attrs_define
class RollupResult:
    """
    Attributes:
        buckets_written (int):
        granularity (Granularity):
    """

    buckets_written: int
    granularity: Granularity
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        buckets_written = self.buckets_written

        granularity = self.granularity.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "buckets_written": buckets_written,
                "granularity": granularity,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        buckets_written = d.pop("buckets_written")

        granularity = Granularity(d.pop("granularity"))

        rollup_result = cls(
            buckets_written=buckets_written,
            granularity=granularity,
        )

        rollup_result.additional_properties = d
        return rollup_result

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
