from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ActivityLatency")


@_attrs_define
class ActivityLatency:
    """
    Attributes:
        leg (str): The leg: stt_latency, the speech-to-text vendor's decode of the transcript that settled the turn;
            llm_ttft, asking the model to its first token; tts_ttfb, the first sentence sent to its first audio; roundtrip,
            the settled transcript to the first audio published.
        turns (int): How many turns the leg was counted for.
        p50_ms (float | None | Unset): The median, in milliseconds. Absent when no turn had the leg.
        p95_ms (float | None | Unset): The 95th percentile, in milliseconds. Absent when no turn had the leg.
        p99_ms (float | None | Unset): The 99th percentile, in milliseconds. Absent when no turn had the leg.
    """

    leg: str
    turns: int
    p50_ms: float | None | Unset = UNSET
    p95_ms: float | None | Unset = UNSET
    p99_ms: float | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        leg = self.leg

        turns = self.turns

        p50_ms: float | None | Unset
        if isinstance(self.p50_ms, Unset):
            p50_ms = UNSET
        else:
            p50_ms = self.p50_ms

        p95_ms: float | None | Unset
        if isinstance(self.p95_ms, Unset):
            p95_ms = UNSET
        else:
            p95_ms = self.p95_ms

        p99_ms: float | None | Unset
        if isinstance(self.p99_ms, Unset):
            p99_ms = UNSET
        else:
            p99_ms = self.p99_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "leg": leg,
                "turns": turns,
            }
        )
        if p50_ms is not UNSET:
            field_dict["p50_ms"] = p50_ms
        if p95_ms is not UNSET:
            field_dict["p95_ms"] = p95_ms
        if p99_ms is not UNSET:
            field_dict["p99_ms"] = p99_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        leg = d.pop("leg")

        turns = d.pop("turns")

        def _parse_p50_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        p50_ms = _parse_p50_ms(d.pop("p50_ms", UNSET))

        def _parse_p95_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        p95_ms = _parse_p95_ms(d.pop("p95_ms", UNSET))

        def _parse_p99_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        p99_ms = _parse_p99_ms(d.pop("p99_ms", UNSET))

        activity_latency = cls(
            leg=leg,
            turns=turns,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
        )

        activity_latency.additional_properties = d
        return activity_latency

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
