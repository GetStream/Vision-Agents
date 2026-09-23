from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.tag_value_summary import TagValueSummary


T = TypeVar("T", bound="TagKeySummary")


@_attrs_define
class TagKeySummary:
    """
    Attributes:
        key (str):  Example: product.
        value_count (int): How many distinct values the key was used with. One means it is context rather than a
            breakdown; hundreds mean it identifies something, such as an end customer, and only its largest values are worth
            a chart.
        cost_micros_total (int):
        request_count (int):
        coverage (float): The share of the window's requests that carry this key, from 0 to 1. A key on half the traffic
            breaks down half the bill, which is worth knowing before it is read as the whole of it.
        top_values (list[TagValueSummary]): The ten largest values, biggest spend first.
    """

    key: str
    value_count: int
    cost_micros_total: int
    request_count: int
    coverage: float
    top_values: list[TagValueSummary]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        key = self.key

        value_count = self.value_count

        cost_micros_total = self.cost_micros_total

        request_count = self.request_count

        coverage = self.coverage

        top_values = []
        for top_values_item_data in self.top_values:
            top_values_item = top_values_item_data.to_dict()
            top_values.append(top_values_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "key": key,
                "value_count": value_count,
                "cost_micros_total": cost_micros_total,
                "request_count": request_count,
                "coverage": coverage,
                "top_values": top_values,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.tag_value_summary import TagValueSummary

        d = dict(src_dict)
        key = d.pop("key")

        value_count = d.pop("value_count")

        cost_micros_total = d.pop("cost_micros_total")

        request_count = d.pop("request_count")

        coverage = d.pop("coverage")

        top_values = []
        _top_values = d.pop("top_values")
        for top_values_item_data in _top_values:
            top_values_item = TagValueSummary.from_dict(top_values_item_data)

            top_values.append(top_values_item)

        tag_key_summary = cls(
            key=key,
            value_count=value_count,
            cost_micros_total=cost_micros_total,
            request_count=request_count,
            coverage=coverage,
            top_values=top_values,
        )

        tag_key_summary.additional_properties = d
        return tag_key_summary

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
