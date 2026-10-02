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
            chunk_schedule (list[int] | Unset): Character counts at which a streaming voice flushes audio. Smaller first
                values start speaking sooner and cost more requests. Live only.
            data_policy (DataPolicy | Unset): What a caller requires of what happens to what they send: the audio they had
                transcribed, or the text they had spoken and the voice speaking it. This is a requirement rather than a
                description: a request naming one is only routed to a model whose declared handling meets it, and if none does
                the request is refused rather than sent somewhere that does not.
            emotion (str | Unset): Affect to speak with, for the providers that take one.
            format_ (str | Unset): Codec, sample rate and bitrate as one name - pcm_16000, mp3_44100_128, ulaw_8000 for
                telephony.
                 Example: pcm_16000.
            languages (list[str] | Unset):
            overwrites (TtsOptionsOverwrites | Unset): Settings for one voice provider that this vocabulary has no word for,
                keyed by provider name, for example {"elevenlabs": {"voice_id": "21m00Tcm4TlvDq8ikWAM"}}. The provider named
                parses its own block and refuses a field it does not have, so an overwrite is either sent or reported rather
                than accepted and dropped. It is also the only way to steer a live voice per vendor, since a voice id from one
                library means nothing at another.
                 Example: {'elevenlabs': {'voice_id': '21m00Tcm4TlvDq8ikWAM'}}.
            pronunciations (TtsOptionsPronunciations | Unset): How to say words the voice gets wrong, keyed by the word.
            providers (list[str] | Unset): A priority list of where to try, in the order given, which wins over target when
                it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, and each is
                expanded where it stands, so the order given is the order tried. Health only moves a provider that is down to
                the back.
                 Example: ['elevenlabs', 'en-low-latency'].
            similarity (float | Unset): How closely a cloned voice tracks its reference.
            speed (float | Unset): Rate of delivery, 1 being the voice's own. Providers differ in the range they accept, so
                one asked for a speed outside its own refuses.
                 Example: 1.
            stability (float | Unset): How much the voice may vary between chunks. Higher is flatter and more consistent.
            style (str | Unset): Delivery style, for the providers that name styles rather than emotions.
            target (str | Unset): A provider/model or a capability shortcut. Example: en-low-latency.
            voice (str | Unset): A provider's own voice id, or one of your voices by id or by the name you gave it. Prefix
                it with custom: to mean only the latter: without the prefix a name that is not one of yours is passed through to
                the provider's library, and with it a name that is not one of yours is refused.
                 Example: custom:receptionist.
            volume (float | Unset): Loudness, 1 being the voice's own.
    """

    chunk_schedule: list[int] | Unset = UNSET
    data_policy: DataPolicy | Unset = UNSET
    emotion: str | Unset = UNSET
    format_: str | Unset = UNSET
    languages: list[str] | Unset = UNSET
    overwrites: TtsOptionsOverwrites | Unset = UNSET
    pronunciations: TtsOptionsPronunciations | Unset = UNSET
    providers: list[str] | Unset = UNSET
    similarity: float | Unset = UNSET
    speed: float | Unset = UNSET
    stability: float | Unset = UNSET
    style: str | Unset = UNSET
    target: str | Unset = UNSET
    voice: str | Unset = UNSET
    volume: float | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        chunk_schedule: list[int] | Unset = UNSET
        if not isinstance(self.chunk_schedule, Unset):
            chunk_schedule = self.chunk_schedule

        data_policy: dict[str, Any] | Unset = UNSET
        if not isinstance(self.data_policy, Unset):
            data_policy = self.data_policy.to_dict()

        emotion = self.emotion

        format_ = self.format_

        languages: list[str] | Unset = UNSET
        if not isinstance(self.languages, Unset):
            languages = self.languages

        overwrites: dict[str, Any] | Unset = UNSET
        if not isinstance(self.overwrites, Unset):
            overwrites = self.overwrites.to_dict()

        pronunciations: dict[str, Any] | Unset = UNSET
        if not isinstance(self.pronunciations, Unset):
            pronunciations = self.pronunciations.to_dict()

        providers: list[str] | Unset = UNSET
        if not isinstance(self.providers, Unset):
            providers = self.providers

        similarity = self.similarity

        speed = self.speed

        stability = self.stability

        style = self.style

        target = self.target

        voice = self.voice

        volume = self.volume

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if chunk_schedule is not UNSET:
            field_dict["chunk_schedule"] = chunk_schedule
        if data_policy is not UNSET:
            field_dict["data_policy"] = data_policy
        if emotion is not UNSET:
            field_dict["emotion"] = emotion
        if format_ is not UNSET:
            field_dict["format"] = format_
        if languages is not UNSET:
            field_dict["languages"] = languages
        if overwrites is not UNSET:
            field_dict["overwrites"] = overwrites
        if pronunciations is not UNSET:
            field_dict["pronunciations"] = pronunciations
        if providers is not UNSET:
            field_dict["providers"] = providers
        if similarity is not UNSET:
            field_dict["similarity"] = similarity
        if speed is not UNSET:
            field_dict["speed"] = speed
        if stability is not UNSET:
            field_dict["stability"] = stability
        if style is not UNSET:
            field_dict["style"] = style
        if target is not UNSET:
            field_dict["target"] = target
        if voice is not UNSET:
            field_dict["voice"] = voice
        if volume is not UNSET:
            field_dict["volume"] = volume

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
        chunk_schedule = cast(list[int], d.pop("chunk_schedule", UNSET))

        _data_policy = d.pop("data_policy", UNSET)
        data_policy: DataPolicy | Unset
        if isinstance(_data_policy, Unset):
            data_policy = UNSET
        else:
            data_policy = DataPolicy.from_dict(_data_policy)

        emotion = d.pop("emotion", UNSET)

        format_ = d.pop("format", UNSET)

        languages = cast(list[str], d.pop("languages", UNSET))

        _overwrites = d.pop("overwrites", UNSET)
        overwrites: TtsOptionsOverwrites | Unset
        if isinstance(_overwrites, Unset):
            overwrites = UNSET
        else:
            overwrites = TtsOptionsOverwrites.from_dict(_overwrites)

        _pronunciations = d.pop("pronunciations", UNSET)
        pronunciations: TtsOptionsPronunciations | Unset
        if isinstance(_pronunciations, Unset):
            pronunciations = UNSET
        else:
            pronunciations = TtsOptionsPronunciations.from_dict(_pronunciations)

        providers = cast(list[str], d.pop("providers", UNSET))

        similarity = d.pop("similarity", UNSET)

        speed = d.pop("speed", UNSET)

        stability = d.pop("stability", UNSET)

        style = d.pop("style", UNSET)

        target = d.pop("target", UNSET)

        voice = d.pop("voice", UNSET)

        volume = d.pop("volume", UNSET)

        tts_options = cls(
            chunk_schedule=chunk_schedule,
            data_policy=data_policy,
            emotion=emotion,
            format_=format_,
            languages=languages,
            overwrites=overwrites,
            pronunciations=pronunciations,
            providers=providers,
            similarity=similarity,
            speed=speed,
            stability=stability,
            style=style,
            target=target,
            voice=voice,
            volume=volume,
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
