from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.endpointing import Endpointing
from ..models.transcript_format import TranscriptFormat
from ..models.transcription_mode import TranscriptionMode
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.data_policy import DataPolicy
    from ..models.stt_options_overwrites import SttOptionsOverwrites


T = TypeVar("T", bound="SttOptions")


@_attrs_define
class SttOptions:
    """How this config transcribes, live or from a recording. A field that only means something on one of the two forms
    says so: a recording has no endpointing to do, and a socket has no file to write subtitles from. A provider that
    cannot express a term refuses the request rather than dropping it silently.

        Attributes:
            channels (int | Unset): Transcribe a multichannel recording per channel rather than mixed down.
            data_policy (DataPolicy | Unset): What a caller requires of what happens to what they send: the audio they had
                transcribed, or the text they had spoken and the voice speaking it. This is a requirement rather than a
                description: a request naming one is only routed to a model whose declared handling meets it, and if none does
                the request is refused rather than sent somewhere that does not.
            detect_language (bool | Unset): Let the provider identify the language instead of being told it.
            diarize (bool | Unset): Label each stretch of speech with who said it.
            eager_end_of_turn (bool | Unset): Send a transcript as soon as the model guesses the turn may be over, before it
                is sure, so a reply can start early. Live only. A model without an eager end of turn transcribes as normal
                rather than being refused. On by default for en-low-latency and multilingual-low-latency.
            endpointing (Endpointing | Unset): What decides a turn is over: a long enough pause, or a model reading the
                words and judging the sentence finished.
            entities (bool | Unset): Extract named entities from the recording. Recording only.
            events (bool | Unset): Tag non-speech audio events such as laughter or music.
            format_ (bool | Unset): Punctuation, capitalisation and smart formatting of numbers and dates.
            interim (bool | Unset): Emit partial transcripts as they firm up, not only final ones. Live only.
            keyterms (list[str] | Unset): Business-specific words the transcriber would otherwise get wrong. Up to 100
                terms, and providers that cannot be told about vocabulary refuse them.
            languages (list[str] | Unset): ISO codes candidates must cover. Empty with detect_language lets the provider
                decide. Example: ['en'].
            max_speakers (int | Unset): A hard cap on the speakers diarization may find, not a hint. Providers differ in
                what they allow, so one asked for more than it supports refuses.
            mode (TranscriptionMode | Unset): How faithfully the transcript follows what was said. verbatim keeps the ums,
                the repetitions and the false starts; smart removes them, tidies the grammar and formats the result, which is
                why it cannot also diarize or time the words - they may no longer be the words that were spoken. Almost no
                provider offers both, so this narrows where a request can go.
            output (TranscriptFormat | Unset): What a finished transcript is rendered as. json carries the words and
                speakers; srt and vtt are subtitle files. Recording only.
            overwrites (SttOptionsOverwrites | Unset): Settings for one provider that this vocabulary has no word for, keyed
                by provider name, for example {"deepgram": {"eot_threshold": 0.6}}. The provider named parses its own block and
                refuses a field it does not have, so an overwrite is either sent or reported rather than accepted and dropped.
                 Example: {'deepgram': {'eot_threshold': 0.6}}.
            profanity_filter (bool | Unset): Mask offensive words rather than writing them down. Only some providers can be
                told to, so a request for it is routed to one of them or refused.
            providers (list[str] | Unset): A priority list of where to try, in the order given, which wins over target when
                it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, and each is
                expanded where it stands, so the order given is the order tried. Health only moves a provider that is down to
                the back; unlike a shortcut, this does not reorder on latency, because a caller who wrote an order meant it.
                 Example: ['deepgram', 'en-low-latency'].
            redact (bool | Unset): Remove personally identifying information from the transcript.
            sample_rate (int | Unset): Rate of the PCM sent on the socket. Zero means 16 kHz. Live only. Example: 16000.
            silence_ms (int | Unset): How long a pause ends a turn, for silence endpointing. Live only. Example: 300.
            summary (bool | Unset): Summarise the recording, where the provider offers audio intelligence. Recording only.
            target (str | Unset): A provider/model or a capability shortcut such as en-low-latency for the live path or en-
                recorded for a recording.
                 Example: en-low-latency.
            utterance_end_ms (int | Unset): How long after the last word an utterance is declared over. Live only.
            words (bool | Unset): Word-level timestamps. Recording only.
    """

    channels: int | Unset = UNSET
    data_policy: DataPolicy | Unset = UNSET
    detect_language: bool | Unset = UNSET
    diarize: bool | Unset = UNSET
    eager_end_of_turn: bool | Unset = UNSET
    endpointing: Endpointing | Unset = UNSET
    entities: bool | Unset = UNSET
    events: bool | Unset = UNSET
    format_: bool | Unset = UNSET
    interim: bool | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    languages: list[str] | Unset = UNSET
    max_speakers: int | Unset = UNSET
    mode: TranscriptionMode | Unset = UNSET
    output: TranscriptFormat | Unset = UNSET
    overwrites: SttOptionsOverwrites | Unset = UNSET
    profanity_filter: bool | Unset = UNSET
    providers: list[str] | Unset = UNSET
    redact: bool | Unset = UNSET
    sample_rate: int | Unset = UNSET
    silence_ms: int | Unset = UNSET
    summary: bool | Unset = UNSET
    target: str | Unset = UNSET
    utterance_end_ms: int | Unset = UNSET
    words: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        channels = self.channels

        data_policy: dict[str, Any] | Unset = UNSET
        if not isinstance(self.data_policy, Unset):
            data_policy = self.data_policy.to_dict()

        detect_language = self.detect_language

        diarize = self.diarize

        eager_end_of_turn = self.eager_end_of_turn

        endpointing: str | Unset = UNSET
        if not isinstance(self.endpointing, Unset):
            endpointing = self.endpointing.value

        entities = self.entities

        events = self.events

        format_ = self.format_

        interim = self.interim

        keyterms: list[str] | Unset = UNSET
        if not isinstance(self.keyterms, Unset):
            keyterms = self.keyterms

        languages: list[str] | Unset = UNSET
        if not isinstance(self.languages, Unset):
            languages = self.languages

        max_speakers = self.max_speakers

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        output: str | Unset = UNSET
        if not isinstance(self.output, Unset):
            output = self.output.value

        overwrites: dict[str, Any] | Unset = UNSET
        if not isinstance(self.overwrites, Unset):
            overwrites = self.overwrites.to_dict()

        profanity_filter = self.profanity_filter

        providers: list[str] | Unset = UNSET
        if not isinstance(self.providers, Unset):
            providers = self.providers

        redact = self.redact

        sample_rate = self.sample_rate

        silence_ms = self.silence_ms

        summary = self.summary

        target = self.target

        utterance_end_ms = self.utterance_end_ms

        words = self.words

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if channels is not UNSET:
            field_dict["channels"] = channels
        if data_policy is not UNSET:
            field_dict["data_policy"] = data_policy
        if detect_language is not UNSET:
            field_dict["detect_language"] = detect_language
        if diarize is not UNSET:
            field_dict["diarize"] = diarize
        if eager_end_of_turn is not UNSET:
            field_dict["eager_end_of_turn"] = eager_end_of_turn
        if endpointing is not UNSET:
            field_dict["endpointing"] = endpointing
        if entities is not UNSET:
            field_dict["entities"] = entities
        if events is not UNSET:
            field_dict["events"] = events
        if format_ is not UNSET:
            field_dict["format"] = format_
        if interim is not UNSET:
            field_dict["interim"] = interim
        if keyterms is not UNSET:
            field_dict["keyterms"] = keyterms
        if languages is not UNSET:
            field_dict["languages"] = languages
        if max_speakers is not UNSET:
            field_dict["max_speakers"] = max_speakers
        if mode is not UNSET:
            field_dict["mode"] = mode
        if output is not UNSET:
            field_dict["output"] = output
        if overwrites is not UNSET:
            field_dict["overwrites"] = overwrites
        if profanity_filter is not UNSET:
            field_dict["profanity_filter"] = profanity_filter
        if providers is not UNSET:
            field_dict["providers"] = providers
        if redact is not UNSET:
            field_dict["redact"] = redact
        if sample_rate is not UNSET:
            field_dict["sample_rate"] = sample_rate
        if silence_ms is not UNSET:
            field_dict["silence_ms"] = silence_ms
        if summary is not UNSET:
            field_dict["summary"] = summary
        if target is not UNSET:
            field_dict["target"] = target
        if utterance_end_ms is not UNSET:
            field_dict["utterance_end_ms"] = utterance_end_ms
        if words is not UNSET:
            field_dict["words"] = words

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.data_policy import DataPolicy
        from ..models.stt_options_overwrites import (
            SttOptionsOverwrites,
        )

        d = dict(src_dict)
        channels = d.pop("channels", UNSET)

        _data_policy = d.pop("data_policy", UNSET)
        data_policy: DataPolicy | Unset
        if isinstance(_data_policy, Unset):
            data_policy = UNSET
        else:
            data_policy = DataPolicy.from_dict(_data_policy)

        detect_language = d.pop("detect_language", UNSET)

        diarize = d.pop("diarize", UNSET)

        eager_end_of_turn = d.pop("eager_end_of_turn", UNSET)

        _endpointing = d.pop("endpointing", UNSET)
        endpointing: Endpointing | Unset
        if isinstance(_endpointing, Unset):
            endpointing = UNSET
        else:
            endpointing = Endpointing(_endpointing)

        entities = d.pop("entities", UNSET)

        events = d.pop("events", UNSET)

        format_ = d.pop("format", UNSET)

        interim = d.pop("interim", UNSET)

        keyterms = cast(list[str], d.pop("keyterms", UNSET))

        languages = cast(list[str], d.pop("languages", UNSET))

        max_speakers = d.pop("max_speakers", UNSET)

        _mode = d.pop("mode", UNSET)
        mode: TranscriptionMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = TranscriptionMode(_mode)

        _output = d.pop("output", UNSET)
        output: TranscriptFormat | Unset
        if isinstance(_output, Unset):
            output = UNSET
        else:
            output = TranscriptFormat(_output)

        _overwrites = d.pop("overwrites", UNSET)
        overwrites: SttOptionsOverwrites | Unset
        if isinstance(_overwrites, Unset):
            overwrites = UNSET
        else:
            overwrites = SttOptionsOverwrites.from_dict(_overwrites)

        profanity_filter = d.pop("profanity_filter", UNSET)

        providers = cast(list[str], d.pop("providers", UNSET))

        redact = d.pop("redact", UNSET)

        sample_rate = d.pop("sample_rate", UNSET)

        silence_ms = d.pop("silence_ms", UNSET)

        summary = d.pop("summary", UNSET)

        target = d.pop("target", UNSET)

        utterance_end_ms = d.pop("utterance_end_ms", UNSET)

        words = d.pop("words", UNSET)

        stt_options = cls(
            channels=channels,
            data_policy=data_policy,
            detect_language=detect_language,
            diarize=diarize,
            eager_end_of_turn=eager_end_of_turn,
            endpointing=endpointing,
            entities=entities,
            events=events,
            format_=format_,
            interim=interim,
            keyterms=keyterms,
            languages=languages,
            max_speakers=max_speakers,
            mode=mode,
            output=output,
            overwrites=overwrites,
            profanity_filter=profanity_filter,
            providers=providers,
            redact=redact,
            sample_rate=sample_rate,
            silence_ms=silence_ms,
            summary=summary,
            target=target,
            utterance_end_ms=utterance_end_ms,
            words=words,
        )

        stt_options.additional_properties = d
        return stt_options

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
