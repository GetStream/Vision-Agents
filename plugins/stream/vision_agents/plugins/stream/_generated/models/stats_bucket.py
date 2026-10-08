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
        audio_ms_total (int): Billable audio, transcribed or produced.
        bucket (datetime.datetime):
        cached_input_tokens_total (int): The part of the prompt served from the provider's cache.
        characters_total (int): Billable text. Zero for providers that bill by audio.
        cost_micros_total (int): Millionths of a dollar, priced from the configured rates.
        error_count (int):
        images_total (int): Pictures drawn. Zero outside image.
        input_tokens_total (int): Prompt tokens read, cached ones included. Zero outside llm.
        model (str):
        output_tokens_total (int): Generated tokens, reasoning included. Zero outside llm.
        provider (str):
        request_count (int):
        latency_p50_ms (float | None | Unset):
        latency_p95_ms (float | None | Unset):
        uptime (float | None | Unset): Successes over total requests in the bucket.
    """

    audio_ms_total: int
    bucket: datetime.datetime
    cached_input_tokens_total: int
    characters_total: int
    cost_micros_total: int
    error_count: int
    images_total: int
    input_tokens_total: int
    model: str
    output_tokens_total: int
    provider: str
    request_count: int
    latency_p50_ms: float | None | Unset = UNSET
    latency_p95_ms: float | None | Unset = UNSET
    uptime: float | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        audio_ms_total = self.audio_ms_total

        bucket = self.bucket.isoformat()

        cached_input_tokens_total = self.cached_input_tokens_total

        characters_total = self.characters_total

        cost_micros_total = self.cost_micros_total

        error_count = self.error_count

        images_total = self.images_total

        input_tokens_total = self.input_tokens_total

        model = self.model

        output_tokens_total = self.output_tokens_total

        provider = self.provider

        request_count = self.request_count

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
                "audio_ms_total": audio_ms_total,
                "bucket": bucket,
                "cached_input_tokens_total": cached_input_tokens_total,
                "characters_total": characters_total,
                "cost_micros_total": cost_micros_total,
                "error_count": error_count,
                "images_total": images_total,
                "input_tokens_total": input_tokens_total,
                "model": model,
                "output_tokens_total": output_tokens_total,
                "provider": provider,
                "request_count": request_count,
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
        audio_ms_total = d.pop("audio_ms_total")

        bucket = datetime.datetime.fromisoformat(d.pop("bucket"))

        cached_input_tokens_total = d.pop("cached_input_tokens_total")

        characters_total = d.pop("characters_total")

        cost_micros_total = d.pop("cost_micros_total")

        error_count = d.pop("error_count")

        images_total = d.pop("images_total")

        input_tokens_total = d.pop("input_tokens_total")

        model = d.pop("model")

        output_tokens_total = d.pop("output_tokens_total")

        provider = d.pop("provider")

        request_count = d.pop("request_count")

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
            audio_ms_total=audio_ms_total,
            bucket=bucket,
            cached_input_tokens_total=cached_input_tokens_total,
            characters_total=characters_total,
            cost_micros_total=cost_micros_total,
            error_count=error_count,
            images_total=images_total,
            input_tokens_total=input_tokens_total,
            model=model,
            output_tokens_total=output_tokens_total,
            provider=provider,
            request_count=request_count,
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
