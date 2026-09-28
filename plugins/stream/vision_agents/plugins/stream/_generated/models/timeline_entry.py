from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_call_timing import ModelCallTiming


T = TypeVar("T", bound="TimelineEntry")


@_attrs_define
class TimelineEntry:
    """
    Attributes:
        turn_id (str):
        started_at (datetime.datetime):
        cadence_ms (float | Unset): Last transcript revision to a stable turn ready for the flow controller.
        decision_ms (float | Unset): Stable turn to the main model request, including flow and queueing.
        model_to_first_text_ms (float | Unset): Main model request to the first text delta admitted to the voice
            pipeline.
        text_to_tts_ms (float | Unset): First text delta to the first TTS request.
        tts_to_audio_ms (float | Unset): First TTS request to the first audio chunk published to the edge.
        model_calls (list[ModelCallTiming] | Unset): Individual model requests for this turn, including flow and
            delegated work.
        heard (str | Unset): What the caller said, when it can be matched to this exchange.
        said (str | Unset): What the agent answered.
        roundtrip_ms (float | Unset): Last transcript revision to first audio published; includes cadence settling.
        stt_latency_ms (float | None | Unset): The provider's decode time for the transcript that settled the turn.
        llm_ttft_ms (float | None | Unset): The wait between asking the model and its first token.
        tts_ttfb_ms (float | None | Unset): The wait between sending the first sentence and the first audio.
        speech_end_to_audio_ms (float | None | Unset): Last input audio to first output audio, estimated using provider
            STT processing time plus roundtrip. It excludes network transport and playback.
        audio_out_ms (float | Unset): How much the agent spoke.
        interrupted (bool | Unset): Whether the caller talked over the answer.
    """

    turn_id: str
    started_at: datetime.datetime
    cadence_ms: float | Unset = UNSET
    decision_ms: float | Unset = UNSET
    model_to_first_text_ms: float | Unset = UNSET
    text_to_tts_ms: float | Unset = UNSET
    tts_to_audio_ms: float | Unset = UNSET
    model_calls: list[ModelCallTiming] | Unset = UNSET
    heard: str | Unset = UNSET
    said: str | Unset = UNSET
    roundtrip_ms: float | Unset = UNSET
    stt_latency_ms: float | None | Unset = UNSET
    llm_ttft_ms: float | None | Unset = UNSET
    tts_ttfb_ms: float | None | Unset = UNSET
    speech_end_to_audio_ms: float | None | Unset = UNSET
    audio_out_ms: float | Unset = UNSET
    interrupted: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        turn_id = self.turn_id

        started_at = self.started_at.isoformat()

        cadence_ms = self.cadence_ms

        decision_ms = self.decision_ms

        model_to_first_text_ms = self.model_to_first_text_ms

        text_to_tts_ms = self.text_to_tts_ms

        tts_to_audio_ms = self.tts_to_audio_ms

        model_calls: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.model_calls, Unset):
            model_calls = []
            for model_calls_item_data in self.model_calls:
                model_calls_item = model_calls_item_data.to_dict()
                model_calls.append(model_calls_item)

        heard = self.heard

        said = self.said

        roundtrip_ms = self.roundtrip_ms

        stt_latency_ms: float | None | Unset
        if isinstance(self.stt_latency_ms, Unset):
            stt_latency_ms = UNSET
        else:
            stt_latency_ms = self.stt_latency_ms

        llm_ttft_ms: float | None | Unset
        if isinstance(self.llm_ttft_ms, Unset):
            llm_ttft_ms = UNSET
        else:
            llm_ttft_ms = self.llm_ttft_ms

        tts_ttfb_ms: float | None | Unset
        if isinstance(self.tts_ttfb_ms, Unset):
            tts_ttfb_ms = UNSET
        else:
            tts_ttfb_ms = self.tts_ttfb_ms

        speech_end_to_audio_ms: float | None | Unset
        if isinstance(self.speech_end_to_audio_ms, Unset):
            speech_end_to_audio_ms = UNSET
        else:
            speech_end_to_audio_ms = self.speech_end_to_audio_ms

        audio_out_ms = self.audio_out_ms

        interrupted = self.interrupted

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "turn_id": turn_id,
                "started_at": started_at,
            }
        )
        if cadence_ms is not UNSET:
            field_dict["cadence_ms"] = cadence_ms
        if decision_ms is not UNSET:
            field_dict["decision_ms"] = decision_ms
        if model_to_first_text_ms is not UNSET:
            field_dict["model_to_first_text_ms"] = model_to_first_text_ms
        if text_to_tts_ms is not UNSET:
            field_dict["text_to_tts_ms"] = text_to_tts_ms
        if tts_to_audio_ms is not UNSET:
            field_dict["tts_to_audio_ms"] = tts_to_audio_ms
        if model_calls is not UNSET:
            field_dict["model_calls"] = model_calls
        if heard is not UNSET:
            field_dict["heard"] = heard
        if said is not UNSET:
            field_dict["said"] = said
        if roundtrip_ms is not UNSET:
            field_dict["roundtrip_ms"] = roundtrip_ms
        if stt_latency_ms is not UNSET:
            field_dict["stt_latency_ms"] = stt_latency_ms
        if llm_ttft_ms is not UNSET:
            field_dict["llm_ttft_ms"] = llm_ttft_ms
        if tts_ttfb_ms is not UNSET:
            field_dict["tts_ttfb_ms"] = tts_ttfb_ms
        if speech_end_to_audio_ms is not UNSET:
            field_dict["speech_end_to_audio_ms"] = speech_end_to_audio_ms
        if audio_out_ms is not UNSET:
            field_dict["audio_out_ms"] = audio_out_ms
        if interrupted is not UNSET:
            field_dict["interrupted"] = interrupted

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.model_call_timing import ModelCallTiming

        d = dict(src_dict)
        turn_id = d.pop("turn_id")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        cadence_ms = d.pop("cadence_ms", UNSET)

        decision_ms = d.pop("decision_ms", UNSET)

        model_to_first_text_ms = d.pop("model_to_first_text_ms", UNSET)

        text_to_tts_ms = d.pop("text_to_tts_ms", UNSET)

        tts_to_audio_ms = d.pop("tts_to_audio_ms", UNSET)

        _model_calls = d.pop("model_calls", UNSET)
        model_calls: list[ModelCallTiming] | Unset = UNSET
        if _model_calls is not UNSET:
            model_calls = []
            for model_calls_item_data in _model_calls:
                model_calls_item = ModelCallTiming.from_dict(model_calls_item_data)

                model_calls.append(model_calls_item)

        heard = d.pop("heard", UNSET)

        said = d.pop("said", UNSET)

        roundtrip_ms = d.pop("roundtrip_ms", UNSET)

        def _parse_stt_latency_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        stt_latency_ms = _parse_stt_latency_ms(d.pop("stt_latency_ms", UNSET))

        def _parse_llm_ttft_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        llm_ttft_ms = _parse_llm_ttft_ms(d.pop("llm_ttft_ms", UNSET))

        def _parse_tts_ttfb_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        tts_ttfb_ms = _parse_tts_ttfb_ms(d.pop("tts_ttfb_ms", UNSET))

        def _parse_speech_end_to_audio_ms(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        speech_end_to_audio_ms = _parse_speech_end_to_audio_ms(
            d.pop("speech_end_to_audio_ms", UNSET)
        )

        audio_out_ms = d.pop("audio_out_ms", UNSET)

        interrupted = d.pop("interrupted", UNSET)

        timeline_entry = cls(
            turn_id=turn_id,
            started_at=started_at,
            cadence_ms=cadence_ms,
            decision_ms=decision_ms,
            model_to_first_text_ms=model_to_first_text_ms,
            text_to_tts_ms=text_to_tts_ms,
            tts_to_audio_ms=tts_to_audio_ms,
            model_calls=model_calls,
            heard=heard,
            said=said,
            roundtrip_ms=roundtrip_ms,
            stt_latency_ms=stt_latency_ms,
            llm_ttft_ms=llm_ttft_ms,
            tts_ttfb_ms=tts_ttfb_ms,
            speech_end_to_audio_ms=speech_end_to_audio_ms,
            audio_out_ms=audio_out_ms,
            interrupted=interrupted,
        )

        timeline_entry.additional_properties = d
        return timeline_entry

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
