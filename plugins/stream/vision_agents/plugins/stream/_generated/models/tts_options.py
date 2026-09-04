from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.tts_options_pronunciations import TtsOptionsPronunciations


T = TypeVar("T", bound="TtsOptions")


@_attrs_define
class TtsOptions:
    """How this config speaks. A provider that cannot express a term refuses the request rather than dropping it silently,
    since a voice asked to sound urgent and speaking flatly is worse than one that says it cannot.

        Attributes:
            target (str | Unset): A provider/model or a capability shortcut. Example: en-low-latency.
            voice (str | Unset): Provider-specific voice id.
            languages (list[str] | Unset):
            speed (float | Unset): Rate of delivery, 1 being the voice's own. Providers differ in the range they accept, so
                one asked for a speed outside its own refuses.
                 Example: 1.
            volume (float | Unset): Loudness, 1 being the voice's own.
            emotion (str | Unset): Affect to speak with, for the providers that take one.
            style (str | Unset): Delivery style, for the providers that name styles rather than emotions.
            stability (float | Unset): How much the voice may vary between chunks. Higher is flatter and more consistent.
            similarity (float | Unset): How closely a cloned voice tracks its reference.
            format_ (str | Unset): Codec, sample rate and bitrate as one name - pcm_16000, mp3_44100_128, ulaw_8000 for
                telephony.
                 Example: pcm_16000.
            pronunciations (TtsOptionsPronunciations | Unset): How to say words the voice gets wrong, keyed by the word.
            chunk_schedule (list[int] | Unset): Character counts at which a streaming voice flushes audio. Smaller first
                values start speaking sooner and cost more requests. Live only.
    """

    target: str | Unset = UNSET
    voice: str | Unset = UNSET
    languages: list[str] | Unset = UNSET
    speed: float | Unset = UNSET
    volume: float | Unset = UNSET
    emotion: str | Unset = UNSET
    style: str | Unset = UNSET
    stability: float | Unset = UNSET
    similarity: float | Unset = UNSET
    format_: str | Unset = UNSET
    pronunciations: TtsOptionsPronunciations | Unset = UNSET
    chunk_schedule: list[int] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        target = self.target

        voice = self.voice

        languages: list[str] | Unset = UNSET
        if not isinstance(self.languages, Unset):
            languages = self.languages

        speed = self.speed

        volume = self.volume

        emotion = self.emotion

        style = self.style

        stability = self.stability

        similarity = self.similarity

        format_ = self.format_

        pronunciations: dict[str, Any] | Unset = UNSET
        if not isinstance(self.pronunciations, Unset):
            pronunciations = self.pronunciations.to_dict()

        chunk_schedule: list[int] | Unset = UNSET
        if not isinstance(self.chunk_schedule, Unset):
            chunk_schedule = self.chunk_schedule

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if target is not UNSET:
            field_dict["target"] = target
        if voice is not UNSET:
            field_dict["voice"] = voice
        if languages is not UNSET:
            field_dict["languages"] = languages
        if speed is not UNSET:
            field_dict["speed"] = speed
        if volume is not UNSET:
            field_dict["volume"] = volume
        if emotion is not UNSET:
            field_dict["emotion"] = emotion
        if style is not UNSET:
            field_dict["style"] = style
        if stability is not UNSET:
            field_dict["stability"] = stability
        if similarity is not UNSET:
            field_dict["similarity"] = similarity
        if format_ is not UNSET:
            field_dict["format"] = format_
        if pronunciations is not UNSET:
            field_dict["pronunciations"] = pronunciations
        if chunk_schedule is not UNSET:
            field_dict["chunk_schedule"] = chunk_schedule

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.tts_options_pronunciations import (
            TtsOptionsPronunciations,
        )

        d = dict(src_dict)
        target = d.pop("target", UNSET)

        voice = d.pop("voice", UNSET)

        languages = cast(list[str], d.pop("languages", UNSET))

        speed = d.pop("speed", UNSET)

        volume = d.pop("volume", UNSET)

        emotion = d.pop("emotion", UNSET)

        style = d.pop("style", UNSET)

        stability = d.pop("stability", UNSET)

        similarity = d.pop("similarity", UNSET)

        format_ = d.pop("format", UNSET)

        _pronunciations = d.pop("pronunciations", UNSET)
        pronunciations: TtsOptionsPronunciations | Unset
        if isinstance(_pronunciations, Unset):
            pronunciations = UNSET
        else:
            pronunciations = TtsOptionsPronunciations.from_dict(_pronunciations)

        chunk_schedule = cast(list[int], d.pop("chunk_schedule", UNSET))

        tts_options = cls(
            target=target,
            voice=voice,
            languages=languages,
            speed=speed,
            volume=volume,
            emotion=emotion,
            style=style,
            stability=stability,
            similarity=similarity,
            format_=format_,
            pronunciations=pronunciations,
            chunk_schedule=chunk_schedule,
        )

        tts_options.additional_properties = d
        return tts_options

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
