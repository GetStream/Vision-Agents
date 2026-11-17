from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="TranscriptWord")


@_attrs_define
class TranscriptWord:
    """
    Attributes:
        end_ms (int):
        start_ms (int):
        text (str):
        confidence (float | Unset):
        speaker (str | Unset): Who said it, when diarization was asked for.
    """

    end_ms: int
    start_ms: int
    text: str
    confidence: float | Unset = UNSET
    speaker: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        end_ms = self.end_ms

        start_ms = self.start_ms

        text = self.text

        confidence = self.confidence

        speaker = self.speaker

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "end_ms": end_ms,
                "start_ms": start_ms,
                "text": text,
            }
        )
        if confidence is not UNSET:
            field_dict["confidence"] = confidence
        if speaker is not UNSET:
            field_dict["speaker"] = speaker

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        end_ms = d.pop("end_ms")

        start_ms = d.pop("start_ms")

        text = d.pop("text")

        confidence = d.pop("confidence", UNSET)

        speaker = d.pop("speaker", UNSET)

        transcript_word = cls(
            end_ms=end_ms,
            start_ms=start_ms,
            text=text,
            confidence=confidence,
            speaker=speaker,
        )

        transcript_word.additional_properties = d
        return transcript_word

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
