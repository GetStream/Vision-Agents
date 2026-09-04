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
        id (str):
        status (RecordingStatus): Where a job has got to. A failed job carries the reason in `error`, and a completed
            one carries its result.
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        provider (str | Unset):
        model (str | Unset):
        format_ (str | Unset): What the audio is encoded as, which is what was asked for. Example: mp3_44100_128.
        url (str | Unset): Where the finished audio is, on a deployment that stores it. Empty means the audio came back
            inline instead.
        audio (str | Unset): The audio itself, base64, when it was not stored behind a URL.
        audio_duration_ms (int | Unset):
        characters (int | Unset): How much text was spoken, which is what it was billed on.
        error (str | Unset):
        completed_at (datetime.datetime | Unset):
    """

    id: str
    status: RecordingStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    provider: str | Unset = UNSET
    model: str | Unset = UNSET
    format_: str | Unset = UNSET
    url: str | Unset = UNSET
    audio: str | Unset = UNSET
    audio_duration_ms: int | Unset = UNSET
    characters: int | Unset = UNSET
    error: str | Unset = UNSET
    completed_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        status = self.status.value

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        provider = self.provider

        model = self.model

        format_ = self.format_

        url = self.url

        audio = self.audio

        audio_duration_ms = self.audio_duration_ms

        characters = self.characters

        error = self.error

        completed_at: str | Unset = UNSET
        if not isinstance(self.completed_at, Unset):
            completed_at = self.completed_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if provider is not UNSET:
            field_dict["provider"] = provider
        if model is not UNSET:
            field_dict["model"] = model
        if format_ is not UNSET:
            field_dict["format"] = format_
        if url is not UNSET:
            field_dict["url"] = url
        if audio is not UNSET:
            field_dict["audio"] = audio
        if audio_duration_ms is not UNSET:
            field_dict["audio_duration_ms"] = audio_duration_ms
        if characters is not UNSET:
            field_dict["characters"] = characters
        if error is not UNSET:
            field_dict["error"] = error
        if completed_at is not UNSET:
            field_dict["completed_at"] = completed_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        id = d.pop("id")

        status = RecordingStatus(d.pop("status"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        provider = d.pop("provider", UNSET)

        model = d.pop("model", UNSET)

        format_ = d.pop("format", UNSET)

        url = d.pop("url", UNSET)

        audio = d.pop("audio", UNSET)

        audio_duration_ms = d.pop("audio_duration_ms", UNSET)

        characters = d.pop("characters", UNSET)

        error = d.pop("error", UNSET)

        _completed_at = d.pop("completed_at", UNSET)
        completed_at: datetime.datetime | Unset
        if isinstance(_completed_at, Unset):
            completed_at = UNSET
        else:
            completed_at = datetime.datetime.fromisoformat(_completed_at)

        speech = cls(
            id=id,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            provider=provider,
            model=model,
            format_=format_,
            url=url,
            audio=audio,
            audio_duration_ms=audio_duration_ms,
            characters=characters,
            error=error,
            completed_at=completed_at,
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
