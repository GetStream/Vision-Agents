from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="TurnStatsBucket")


@_attrs_define
class TurnStatsBucket:
    """
    Attributes:
        agent_id (str):
        bucket (datetime.datetime):
        turn_count (int):
        interrupted_count (int): Turns a participant talked over before they finished.
        audio_out_ms_total (float): How much speech the agent published in the bucket.
        stt_latency_p50_ms (float | None | Unset):
        stt_latency_p95_ms (float | None | Unset):
        llm_ttft_p50_ms (float | None | Unset):
        llm_ttft_p95_ms (float | None | Unset):
        tts_ttfb_p50_ms (float | None | Unset):
        tts_ttfb_p95_ms (float | None | Unset):
        roundtrip_p50_ms (float | None | Unset): Settled transcript to first audio published.
        roundtrip_p95_ms (float | None | Unset):
        roundtrip_p99_ms (float | None | Unset):
    """

    agent_id: str
    bucket: datetime.datetime
    turn_count: int
    interrupted_count: int
    audio_out_ms_total: float
    stt_latency_p50_ms: float | None | Unset = UNSET
    stt_latency_p95_ms: float | None | Unset = UNSET
    llm_ttft_p50_ms: float | None | Unset = UNSET
    llm_ttft_p95_ms: float | None | Unset = UNSET
    tts_ttfb_p50_ms: float | None | Unset = UNSET
    tts_ttfb_p95_ms: float | None | Unset = UNSET
    roundtrip_p50_ms: float | None | Unset = UNSET
    roundtrip_p95_ms: float | None | Unset = UNSET
    roundtrip_p99_ms: float | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent_id = self.agent_id

        bucket = self.bucket.isoformat()

        turn_count = self.turn_count

        interrupted_count = self.interrupted_count

        audio_out_ms_total = self.audio_out_ms_total

        stt_latency_p50_ms: float | None | Unset
        if isinstance(self.stt_latency_p50_ms, Unset):
            stt_latency_p50_ms = UNSET
        else:
            stt_latency_p50_ms = self.stt_latency_p50_ms

        stt_latency_p95_ms: float | None | Unset
        if isinstance(self.stt_latency_p95_ms, Unset):
            stt_latency_p95_ms = UNSET
        else:
            stt_latency_p95_ms = self.stt_latency_p95_ms

        llm_ttft_p50_ms: float | None | Unset
        if isinstance(self.llm_ttft_p50_ms, Unset):
            llm_ttft_p50_ms = UNSET
        else:
            llm_ttft_p50_ms = self.llm_ttft_p50_ms

        llm_ttft_p95_ms: float | None | Unset
        if isinstance(self.llm_ttft_p95_ms, Unset):
            llm_ttft_p95_ms = UNSET
        else:
            llm_ttft_p95_ms = self.llm_ttft_p95_ms

        tts_ttfb_p50_ms: float | None | Unset
        if isinstance(self.tts_ttfb_p50_ms, Unset):
            tts_ttfb_p50_ms = UNSET
        else:
            tts_ttfb_p50_ms = self.tts_ttfb_p50_ms

        tts_ttfb_p95_ms: float | None | Unset
        if isinstance(self.tts_ttfb_p95_ms, Unset):
            tts_ttfb_p95_ms = UNSET
        else:
            tts_ttfb_p95_ms = self.tts_ttfb_p95_ms

        roundtrip_p50_ms: float | None | Unset
        if isinstance(self.roundtrip_p50_ms, Unset):
            roundtrip_p50_ms = UNSET
        else:
            roundtrip_p50_ms = self.roundtrip_p50_ms

        roundtrip_p95_ms: float | None | Unset
        if isinstance(self.roundtrip_p95_ms, Unset):
            roundtrip_p95_ms = UNSET
        else:
            roundtrip_p95_ms = self.roundtrip_p95_ms

        roundtrip_p99_ms: float | None | Unset
        if isinstance(self.roundtrip_p99_ms, Unset):
            roundtrip_p99_ms = UNSET
        else:
            roundtrip_p99_ms = self.roundtrip_p99_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "agent_id": agent_id,
                "bucket": bucket,
                "turn_count": turn_count,
                "interrupted_count": interrupted_count,
                "audio_out_ms_total": audio_out_ms_total,
            }
        )
        if stt_latency_p50_ms is not UNSET:
            field_dict["stt_latency_p50_ms"] = stt_latency_p50_ms
        if stt_latency_p95_ms is not UNSET:
            field_dict["stt_latency_p95_ms"] = stt_latency_p95_ms
        if llm_ttft_p50_ms is not UNSET:
            field_dict["llm_ttft_p50_ms"] = llm_ttft_p50_ms
        if llm_ttft_p95_ms is not UNSET:
            field_dict["llm_ttft_p95_ms"] = llm_ttft_p95_ms
        if tts_ttfb_p50_ms is not UNSET:
            field_dict["tts_ttfb_p50_ms"] = tts_ttfb_p50_ms
        if tts_ttfb_p95_ms is not UNSET:
            field_dict["tts_ttfb_p95_ms"] = tts_ttfb_p95_ms
        if roundtrip_p50_ms is not UNSET:
            field_dict["roundtrip_p50_ms"] = roundtrip_p50_ms
        if roundtrip_p95_ms is not UNSET:
            field_dict["roundtrip_p95_ms"] = roundtrip_p95_ms
        if roundtrip_p99_ms is not UNSET:
            field_dict["roundtrip_p99_ms"] = roundtrip_p99_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        agent_id = d.pop("agent_id")

        bucket = datetime.datetime.fromisoformat(d.pop("bucket"))

        turn_count = d.pop("turn_count")

        interrupted_count = d.pop("interrupted_count")

        audio_out_ms_total = d.pop("audio_out_ms_total")

        def _parse_stt_latency_p50_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        stt_latency_p50_ms = _parse_stt_latency_p50_ms(
            d.pop("stt_latency_p50_ms", UNSET)
        )

        def _parse_stt_latency_p95_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        stt_latency_p95_ms = _parse_stt_latency_p95_ms(
            d.pop("stt_latency_p95_ms", UNSET)
        )

        def _parse_llm_ttft_p50_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        llm_ttft_p50_ms = _parse_llm_ttft_p50_ms(d.pop("llm_ttft_p50_ms", UNSET))

        def _parse_llm_ttft_p95_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        llm_ttft_p95_ms = _parse_llm_ttft_p95_ms(d.pop("llm_ttft_p95_ms", UNSET))

        def _parse_tts_ttfb_p50_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        tts_ttfb_p50_ms = _parse_tts_ttfb_p50_ms(d.pop("tts_ttfb_p50_ms", UNSET))

        def _parse_tts_ttfb_p95_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        tts_ttfb_p95_ms = _parse_tts_ttfb_p95_ms(d.pop("tts_ttfb_p95_ms", UNSET))

        def _parse_roundtrip_p50_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        roundtrip_p50_ms = _parse_roundtrip_p50_ms(d.pop("roundtrip_p50_ms", UNSET))

        def _parse_roundtrip_p95_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        roundtrip_p95_ms = _parse_roundtrip_p95_ms(d.pop("roundtrip_p95_ms", UNSET))

        def _parse_roundtrip_p99_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        roundtrip_p99_ms = _parse_roundtrip_p99_ms(d.pop("roundtrip_p99_ms", UNSET))

        turn_stats_bucket = cls(
            agent_id=agent_id,
            bucket=bucket,
            turn_count=turn_count,
            interrupted_count=interrupted_count,
            audio_out_ms_total=audio_out_ms_total,
            stt_latency_p50_ms=stt_latency_p50_ms,
            stt_latency_p95_ms=stt_latency_p95_ms,
            llm_ttft_p50_ms=llm_ttft_p50_ms,
            llm_ttft_p95_ms=llm_ttft_p95_ms,
            tts_ttfb_p50_ms=tts_ttfb_p50_ms,
            tts_ttfb_p95_ms=tts_ttfb_p95_ms,
            roundtrip_p50_ms=roundtrip_p50_ms,
            roundtrip_p95_ms=roundtrip_p95_ms,
            roundtrip_p99_ms=roundtrip_p99_ms,
        )

        turn_stats_bucket.additional_properties = d
        return turn_stats_bucket

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
