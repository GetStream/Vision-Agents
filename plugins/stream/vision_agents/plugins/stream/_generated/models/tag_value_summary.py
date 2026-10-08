from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="TagValueSummary")


@_attrs_define
class TagValueSummary:
    """
    Attributes:
        cost_micros_total (int):
        request_count (int):
        share (float): This value's share of what the key covers, from 0 to 1.
        value (str):  Example: support.
    """

    cost_micros_total: int
    request_count: int
    share: float
    value: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cost_micros_total = self.cost_micros_total

        request_count = self.request_count

        share = self.share

        value = self.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "cost_micros_total": cost_micros_total,
                "request_count": request_count,
                "share": share,
                "value": value,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        cost_micros_total = d.pop("cost_micros_total")

        request_count = d.pop("request_count")

        share = d.pop("share")

        value = d.pop("value")

        tag_value_summary = cls(
            cost_micros_total=cost_micros_total,
            request_count=request_count,
            share=share,
            value=value,
        )

        tag_value_summary.additional_properties = d
        return tag_value_summary

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
