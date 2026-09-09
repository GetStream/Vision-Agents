from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.recording_status import RecordingStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.transcript_entity import TranscriptEntity
    from ..models.transcript_word import TranscriptWord


T = TypeVar("T", bound="Transcription")


@_attrs_define
class Transcription:
    """
    Attributes:
        id (str):
        status (RecordingStatus): Where a job has got to. A failed job carries the reason in `error`, and a completed
            one carries its result.
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        provider (str | Unset):
        model (str | Unset):
        language (str | Unset): What was spoken, whether it was asked for or detected.
        text (str | Unset): The whole transcript as prose.
        words (list[TranscriptWord] | Unset): Present when word-level timestamps were asked for.
        speakers (list[str] | Unset): The speakers diarization found, in the order they first spoke.
        subtitles (str | Unset): The transcript as an SRT or VTT file, when one of those was asked for.
        summary (str | Unset):
        entities (list[TranscriptEntity] | Unset):
        audio_duration_ms (int | Unset): How long the recording was, which is what it was billed on.
        error (str | Unset): Why the job failed, if it did.
        completed_at (datetime.datetime | Unset):
    """

    id: str
    status: RecordingStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    provider: str | Unset = UNSET
    model: str | Unset = UNSET
    language: str | Unset = UNSET
    text: str | Unset = UNSET
    words: list[TranscriptWord] | Unset = UNSET
    speakers: list[str] | Unset = UNSET
    subtitles: str | Unset = UNSET
    summary: str | Unset = UNSET
    entities: list[TranscriptEntity] | Unset = UNSET
    audio_duration_ms: int | Unset = UNSET
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

        language = self.language

        text = self.text

        words: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.words, Unset):
            words = []
            for words_item_data in self.words:
                words_item = words_item_data.to_dict()
                words.append(words_item)

        speakers: list[str] | Unset = UNSET
        if not isinstance(self.speakers, Unset):
            speakers = self.speakers

        subtitles = self.subtitles

        summary = self.summary

        entities: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.entities, Unset):
            entities = []
            for entities_item_data in self.entities:
                entities_item = entities_item_data.to_dict()
                entities.append(entities_item)

        audio_duration_ms = self.audio_duration_ms

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
        if language is not UNSET:
            field_dict["language"] = language
        if text is not UNSET:
            field_dict["text"] = text
        if words is not UNSET:
            field_dict["words"] = words
        if speakers is not UNSET:
            field_dict["speakers"] = speakers
        if subtitles is not UNSET:
            field_dict["subtitles"] = subtitles
        if summary is not UNSET:
            field_dict["summary"] = summary
        if entities is not UNSET:
            field_dict["entities"] = entities
        if audio_duration_ms is not UNSET:
            field_dict["audio_duration_ms"] = audio_duration_ms
        if error is not UNSET:
            field_dict["error"] = error
        if completed_at is not UNSET:
            field_dict["completed_at"] = completed_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.transcript_entity import TranscriptEntity
        from ..models.transcript_word import TranscriptWord

        d = dict(src_dict)
        id = d.pop("id")

        status = RecordingStatus(d.pop("status"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        provider = d.pop("provider", UNSET)

        model = d.pop("model", UNSET)

        language = d.pop("language", UNSET)

        text = d.pop("text", UNSET)

        _words = d.pop("words", UNSET)
        words: list[TranscriptWord] | Unset = UNSET
        if _words is not UNSET:
            words = []
            for words_item_data in _words:
                words_item = TranscriptWord.from_dict(words_item_data)

                words.append(words_item)

        speakers = cast(list[str], d.pop("speakers", UNSET))

        subtitles = d.pop("subtitles", UNSET)

        summary = d.pop("summary", UNSET)

        _entities = d.pop("entities", UNSET)
        entities: list[TranscriptEntity] | Unset = UNSET
        if _entities is not UNSET:
            entities = []
            for entities_item_data in _entities:
                entities_item = TranscriptEntity.from_dict(entities_item_data)

                entities.append(entities_item)

        audio_duration_ms = d.pop("audio_duration_ms", UNSET)

        error = d.pop("error", UNSET)

        _completed_at = d.pop("completed_at", UNSET)
        completed_at: datetime.datetime | Unset
        if isinstance(_completed_at, Unset):
            completed_at = UNSET
        else:
            completed_at = datetime.datetime.fromisoformat(_completed_at)

        transcription = cls(
            id=id,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            provider=provider,
            model=model,
            language=language,
            text=text,
            words=words,
            speakers=speakers,
            subtitles=subtitles,
            summary=summary,
            entities=entities,
            audio_duration_ms=audio_duration_ms,
            error=error,
            completed_at=completed_at,
        )

        transcription.additional_properties = d
        return transcription

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
