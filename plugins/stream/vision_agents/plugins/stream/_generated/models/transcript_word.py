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
        text (str):
        start_ms (int):
        end_ms (int):
        confidence (float | Unset):
        speaker (str | Unset): Who said it, when diarization was asked for.
    """

    text: str
    start_ms: int
    end_ms: int
    confidence: float | Unset = UNSET
    speaker: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        start_ms = self.start_ms

        end_ms = self.end_ms

        confidence = self.confidence

        speaker = self.speaker

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
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
        text = d.pop("text")

        start_ms = d.pop("start_ms")

        end_ms = d.pop("end_ms")

        confidence = d.pop("confidence", UNSET)

        speaker = d.pop("speaker", UNSET)

        transcript_word = cls(
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
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
