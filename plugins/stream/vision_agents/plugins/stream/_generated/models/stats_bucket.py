from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="StatsBucket")


@_attrs_define
class StatsBucket:
    """
    Attributes:
        provider (str):
        model (str):
        bucket (datetime.datetime):
        audio_ms_total (int): Billable audio, transcribed or produced.
        characters_total (int): Billable text. Zero for providers that bill by audio.
        input_tokens_total (int): Prompt tokens read, cached ones included. Zero outside llm.
        cached_input_tokens_total (int): The part of the prompt served from the provider's cache.
        output_tokens_total (int): Generated tokens, reasoning included. Zero outside llm.
        cost_micros_total (int): Millionths of a dollar, priced from the configured rates.
        request_count (int):
        error_count (int):
        latency_p50_ms (float | None | Unset):
        latency_p95_ms (float | None | Unset):
        uptime (float | None | Unset): Successes over total requests in the bucket.
    """

    provider: str
    model: str
    bucket: datetime.datetime
    audio_ms_total: int
    characters_total: int
    input_tokens_total: int
    cached_input_tokens_total: int
    output_tokens_total: int
    cost_micros_total: int
    request_count: int
    error_count: int
    latency_p50_ms: float | None | Unset = UNSET
    latency_p95_ms: float | None | Unset = UNSET
    uptime: float | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider

        model = self.model

        bucket = self.bucket.isoformat()

        audio_ms_total = self.audio_ms_total

        characters_total = self.characters_total

        input_tokens_total = self.input_tokens_total

        cached_input_tokens_total = self.cached_input_tokens_total

        output_tokens_total = self.output_tokens_total

        cost_micros_total = self.cost_micros_total

        request_count = self.request_count

        error_count = self.error_count

        latency_p50_ms: float | None | Unset
        if isinstance(self.latency_p50_ms, Unset):
            latency_p50_ms = UNSET
        else:
            latency_p50_ms = self.latency_p50_ms

        latency_p95_ms: float | None | Unset
        if isinstance(self.latency_p95_ms, Unset):
            latency_p95_ms = UNSET
        else:
            latency_p95_ms = self.latency_p95_ms

        uptime: float | None | Unset
        if isinstance(self.uptime, Unset):
            uptime = UNSET
        else:
            uptime = self.uptime

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "provider": provider,
                "model": model,
                "bucket": bucket,
                "audio_ms_total": audio_ms_total,
                "characters_total": characters_total,
                "input_tokens_total": input_tokens_total,
                "cached_input_tokens_total": cached_input_tokens_total,
                "output_tokens_total": output_tokens_total,
                "cost_micros_total": cost_micros_total,
                "request_count": request_count,
                "error_count": error_count,
            }
        )
        if latency_p50_ms is not UNSET:
            field_dict["latency_p50_ms"] = latency_p50_ms
        if latency_p95_ms is not UNSET:
            field_dict["latency_p95_ms"] = latency_p95_ms
        if uptime is not UNSET:
            field_dict["uptime"] = uptime

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        provider = d.pop("provider")

        model = d.pop("model")

        bucket = datetime.datetime.fromisoformat(d.pop("bucket"))

        audio_ms_total = d.pop("audio_ms_total")

        characters_total = d.pop("characters_total")

        input_tokens_total = d.pop("input_tokens_total")

        cached_input_tokens_total = d.pop("cached_input_tokens_total")

        output_tokens_total = d.pop("output_tokens_total")

        cost_micros_total = d.pop("cost_micros_total")

        request_count = d.pop("request_count")

        error_count = d.pop("error_count")

        def _parse_latency_p50_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        latency_p50_ms = _parse_latency_p50_ms(d.pop("latency_p50_ms", UNSET))

        def _parse_latency_p95_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        latency_p95_ms = _parse_latency_p95_ms(d.pop("latency_p95_ms", UNSET))

        def _parse_uptime(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        uptime = _parse_uptime(d.pop("uptime", UNSET))

        stats_bucket = cls(
            provider=provider,
            model=model,
            bucket=bucket,
            audio_ms_total=audio_ms_total,
            characters_total=characters_total,
            input_tokens_total=input_tokens_total,
            cached_input_tokens_total=cached_input_tokens_total,
            output_tokens_total=output_tokens_total,
            cost_micros_total=cost_micros_total,
            request_count=request_count,
            error_count=error_count,
            latency_p50_ms=latency_p50_ms,
            latency_p95_ms=latency_p95_ms,
            uptime=uptime,
        )

        stats_bucket.additional_properties = d
        return stats_bucket

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
