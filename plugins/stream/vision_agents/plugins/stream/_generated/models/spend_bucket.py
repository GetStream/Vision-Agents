from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="SpendBucket")


@_attrs_define
class SpendBucket:
    """
    Attributes:
        bucket (datetime.datetime):
        value (str): The modality or label value this row is for. "other" is everything outside the biggest few, and the
            empty string is spend carrying no such label at all, so a customer that labels only part of its traffic can see
            which part.
             Example: support.
        cost_micros_total (int): Millionths of a dollar, priced from the configured rates.
        request_count (int):
    """

    bucket: datetime.datetime
    value: str
    cost_micros_total: int
    request_count: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        bucket = self.bucket.isoformat()

        value = self.value

        cost_micros_total = self.cost_micros_total

        request_count = self.request_count

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "bucket": bucket,
                "value": value,
                "cost_micros_total": cost_micros_total,
                "request_count": request_count,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        bucket = datetime.datetime.fromisoformat(d.pop("bucket"))

        value = d.pop("value")

        cost_micros_total = d.pop("cost_micros_total")

        request_count = d.pop("request_count")

        spend_bucket = cls(
            bucket=bucket,
            value=value,
            cost_micros_total=cost_micros_total,
            request_count=request_count,
        )

        spend_bucket.additional_properties = d
        return spend_bucket

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
