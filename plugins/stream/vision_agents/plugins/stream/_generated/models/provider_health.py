from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ProviderHealth")


@_attrs_define
class ProviderHealth:
    """
    Attributes:
        available (bool): False once the error rate crosses the configured threshold.
        requests (int): Requests seen in the current health window.
        errors (int):
        error_rate (float):
        latency_ms_avg (float):
    """

    available: bool
    requests: int
    errors: int
    error_rate: float
    latency_ms_avg: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        available = self.available

        requests = self.requests

        errors = self.errors

        error_rate = self.error_rate

        latency_ms_avg = self.latency_ms_avg

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "available": available,
                "requests": requests,
                "errors": errors,
                "error_rate": error_rate,
                "latency_ms_avg": latency_ms_avg,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        available = d.pop("available")

        requests = d.pop("requests")

        errors = d.pop("errors")

        error_rate = d.pop("error_rate")

        latency_ms_avg = d.pop("latency_ms_avg")

        provider_health = cls(
            available=available,
            requests=requests,
            errors=errors,
            error_rate=error_rate,
            latency_ms_avg=latency_ms_avg,
        )

        provider_health.additional_properties = d
        return provider_health

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
