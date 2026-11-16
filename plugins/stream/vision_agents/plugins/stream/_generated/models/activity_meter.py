from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ActivityMeter")


@_attrs_define
class ActivityMeter:
    """
    Attributes:
        audio_ms (int): Billable audio, transcribed or produced, in milliseconds.
        characters (int): Billable text, in characters.
        cost_micros (int): What it is charged, in millionths of a dollar.
        duration_ms (int): How long the work took, in milliseconds: for runtime and sandbox, the time held.
        input_tokens (int): Tokens read, cached ones included.
        meter (str): The modality, such as llm, tts, runtime or sandbox.
        output_tokens (int): Tokens generated.
        requests (int): How many requests. Runtime and sandbox are time, not requests, and read 0.
    """

    audio_ms: int
    characters: int
    cost_micros: int
    duration_ms: int
    input_tokens: int
    meter: str
    output_tokens: int
    requests: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        audio_ms = self.audio_ms

        characters = self.characters

        cost_micros = self.cost_micros

        duration_ms = self.duration_ms

        input_tokens = self.input_tokens

        meter = self.meter

        output_tokens = self.output_tokens

        requests = self.requests

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "audio_ms": audio_ms,
                "characters": characters,
                "cost_micros": cost_micros,
                "duration_ms": duration_ms,
                "input_tokens": input_tokens,
                "meter": meter,
                "output_tokens": output_tokens,
                "requests": requests,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        audio_ms = d.pop("audio_ms")

        characters = d.pop("characters")

        cost_micros = d.pop("cost_micros")

        duration_ms = d.pop("duration_ms")

        input_tokens = d.pop("input_tokens")

        meter = d.pop("meter")

        output_tokens = d.pop("output_tokens")

        requests = d.pop("requests")

        activity_meter = cls(
            audio_ms=audio_ms,
            characters=characters,
            cost_micros=cost_micros,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            meter=meter,
            output_tokens=output_tokens,
            requests=requests,
        )

        activity_meter.additional_properties = d
        return activity_meter

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
