from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.recording_status import RecordingStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="Speech")


@_attrs_define
class Speech:
    """
    Attributes:
        created_at (datetime.datetime):
        id (str):
        status (RecordingStatus): Where a job has got to. A failed job carries the reason in `error`, and a completed
            one carries its result.
        updated_at (datetime.datetime):
        audio (str | Unset): The audio itself, base64, when it was not stored behind a URL.
        audio_duration_ms (int | Unset):
        characters (int | Unset): How much text was spoken, which is what it was billed on.
        completed_at (datetime.datetime | Unset):
        error (str | Unset):
        format_ (str | Unset): What the audio is encoded as, which is what was asked for. Example: mp3_44100_128.
        model (str | Unset):
        provider (str | Unset):
        url (str | Unset): Where the finished audio is, on a deployment that stores it. Empty means the audio came back
            inline instead.
    """

    created_at: datetime.datetime
    id: str
    status: RecordingStatus
    updated_at: datetime.datetime
    audio: str | Unset = UNSET
    audio_duration_ms: int | Unset = UNSET
    characters: int | Unset = UNSET
    completed_at: datetime.datetime | Unset = UNSET
    error: str | Unset = UNSET
    format_: str | Unset = UNSET
    model: str | Unset = UNSET
    provider: str | Unset = UNSET
    url: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        id = self.id

        status = self.status.value

        updated_at = self.updated_at.isoformat()

        audio = self.audio

        audio_duration_ms = self.audio_duration_ms

        characters = self.characters

        completed_at: str | Unset = UNSET
        if not isinstance(self.completed_at, Unset):
            completed_at = self.completed_at.isoformat()

        error = self.error

        format_ = self.format_

        model = self.model

        provider = self.provider

        url = self.url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "id": id,
                "status": status,
                "updated_at": updated_at,
            }
        )
        if audio is not UNSET:
            field_dict["audio"] = audio
        if audio_duration_ms is not UNSET:
            field_dict["audio_duration_ms"] = audio_duration_ms
        if characters is not UNSET:
            field_dict["characters"] = characters
        if completed_at is not UNSET:
            field_dict["completed_at"] = completed_at
        if error is not UNSET:
            field_dict["error"] = error
        if format_ is not UNSET:
            field_dict["format"] = format_
        if model is not UNSET:
            field_dict["model"] = model
        if provider is not UNSET:
            field_dict["provider"] = provider
        if url is not UNSET:
            field_dict["url"] = url

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        status = RecordingStatus(d.pop("status"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        audio = d.pop("audio", UNSET)

        audio_duration_ms = d.pop("audio_duration_ms", UNSET)

        characters = d.pop("characters", UNSET)

        _completed_at = d.pop("completed_at", UNSET)
        completed_at: datetime.datetime | Unset
        if isinstance(_completed_at, Unset):
            completed_at = UNSET
        else:
            completed_at = datetime.datetime.fromisoformat(_completed_at)

        error = d.pop("error", UNSET)

        format_ = d.pop("format", UNSET)

        model = d.pop("model", UNSET)

        provider = d.pop("provider", UNSET)

        url = d.pop("url", UNSET)

        speech = cls(
            created_at=created_at,
            id=id,
            status=status,
            updated_at=updated_at,
            audio=audio,
            audio_duration_ms=audio_duration_ms,
            characters=characters,
            completed_at=completed_at,
            error=error,
            format_=format_,
            model=model,
            provider=provider,
            url=url,
        )

        speech.additional_properties = d
        return speech

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
