from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.data_policy import DataPolicy
    from ..models.tts_options_overwrites import TtsOptionsOverwrites
    from ..models.tts_options_pronunciations import TtsOptionsPronunciations


T = TypeVar("T", bound="TtsOptions")


@_attrs_define
class TtsOptions:
    """How this config speaks. A provider that cannot express a term refuses the request rather than dropping it silently,
    since a voice asked to sound urgent and speaking flatly is worse than one that says it cannot.

        Attributes:
            target (str | Unset): A provider/model or a capability shortcut. Example: en-low-latency.
            providers (list[str] | Unset): A priority list of where to try, in the order given, which wins over target when
                it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, and each is
                expanded where it stands, so the order given is the order tried. Health only moves a provider that is down to
                the back.
                 Example: ['elevenlabs', 'en-low-latency'].
            voice (str | Unset): A provider's own voice id, or one of your voices by id or by the name you gave it. Prefix
                it with custom: to mean only the latter: without the prefix a name that is not one of yours is passed through to
                the provider's library, and with it a name that is not one of yours is refused.
                 Example: custom:receptionist.
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
            data_policy (DataPolicy | Unset): What a caller requires of what happens to what they send: the audio they had
                transcribed, or the text they had spoken and the voice speaking it. This is a requirement rather than a
                description: a request naming one is only routed to a model whose declared handling meets it, and if none does
                the request is refused rather than sent somewhere that does not.
            overwrites (TtsOptionsOverwrites | Unset): Settings for one voice provider that this vocabulary has no word for,
                keyed by provider name, for example {"elevenlabs": {"voice_id": "21m00Tcm4TlvDq8ikWAM"}}. The provider named
                parses its own block and refuses a field it does not have, so an overwrite is either sent or reported rather
                than accepted and dropped. It is also the only way to steer a live voice per vendor, since a voice id from one
                library means nothing at another.
                 Example: {'elevenlabs': {'voice_id': '21m00Tcm4TlvDq8ikWAM'}}.
    """

    target: str | Unset = UNSET
    providers: list[str] | Unset = UNSET
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
    data_policy: DataPolicy | Unset = UNSET
    overwrites: TtsOptionsOverwrites | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        target = self.target

        providers: list[str] | Unset = UNSET
        if not isinstance(self.providers, Unset):
            providers = self.providers

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

        data_policy: dict[str, Any] | Unset = UNSET
        if not isinstance(self.data_policy, Unset):
            data_policy = self.data_policy.to_dict()

        overwrites: dict[str, Any] | Unset = UNSET
        if not isinstance(self.overwrites, Unset):
            overwrites = self.overwrites.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if target is not UNSET:
            field_dict["target"] = target
        if providers is not UNSET:
            field_dict["providers"] = providers
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
        if data_policy is not UNSET:
            field_dict["data_policy"] = data_policy
        if overwrites is not UNSET:
            field_dict["overwrites"] = overwrites

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.data_policy import DataPolicy
        from ..models.tts_options_overwrites import (
            TtsOptionsOverwrites,
        )
        from ..models.tts_options_pronunciations import (
            TtsOptionsPronunciations,
        )

        d = dict(src_dict)
        target = d.pop("target", UNSET)

        providers = cast(list[str], d.pop("providers", UNSET))

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

        _data_policy = d.pop("data_policy", UNSET)
        data_policy: DataPolicy | Unset
        if isinstance(_data_policy, Unset):
            data_policy = UNSET
        else:
            data_policy = DataPolicy.from_dict(_data_policy)

        _overwrites = d.pop("overwrites", UNSET)
        overwrites: TtsOptionsOverwrites | Unset
        if isinstance(_overwrites, Unset):
            overwrites = UNSET
        else:
            overwrites = TtsOptionsOverwrites.from_dict(_overwrites)

        tts_options = cls(
            target=target,
            providers=providers,
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
            data_policy=data_policy,
            overwrites=overwrites,
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
