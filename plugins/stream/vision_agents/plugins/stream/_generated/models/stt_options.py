from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.endpointing import Endpointing
from ..models.transcript_format import TranscriptFormat
from ..types import UNSET, Unset

T = TypeVar("T", bound="SttOptions")


@_attrs_define
class SttOptions:
    """How this config transcribes, live or from a recording. A field that only means something on one of the two forms
    says so: a recording has no endpointing to do, and a socket has no file to write subtitles from. A provider that
    cannot express a term refuses the request rather than dropping it silently.

        Attributes:
            target (str | Unset): A provider/model or a capability shortcut such as en-low-latency for the live path or en-
                recorded for a recording.
                 Example: en-low-latency.
            languages (list[str] | Unset): ISO codes candidates must cover. Empty with detect_language lets the provider
                decide. Example: ['en'].
            detect_language (bool | Unset): Let the provider identify the language instead of being told it.
            sample_rate (int | Unset): Rate of the PCM sent on the socket. Zero means 16 kHz. Live only. Example: 16000.
            interim (bool | Unset): Emit partial transcripts as they firm up, not only final ones. Live only.
            endpointing (Endpointing | Unset): What decides a turn is over: a long enough pause, or a model reading the
                words and judging the sentence finished.
            silence_ms (int | Unset): How long a pause ends a turn, for silence endpointing. Live only. Example: 300.
            utterance_end_ms (int | Unset): How long after the last word an utterance is declared over. Live only.
            diarize (bool | Unset): Label each stretch of speech with who said it.
            max_speakers (int | Unset): A hard cap on the speakers diarization may find, not a hint. Providers differ in
                what they allow, so one asked for more than it supports refuses.
            keyterms (list[str] | Unset): Business-specific words the transcriber would otherwise get wrong. Up to 100
                terms, and providers that cannot be told about vocabulary refuse them.
            format_ (bool | Unset): Punctuation, capitalisation and smart formatting of numbers and dates.
            redact (bool | Unset): Remove personally identifying information from the transcript.
            events (bool | Unset): Tag non-speech audio events such as laughter or music.
            channels (int | Unset): Transcribe a multichannel recording per channel rather than mixed down.
            words (bool | Unset): Word-level timestamps. Recording only.
            output (TranscriptFormat | Unset): What a finished transcript is rendered as. json carries the words and
                speakers; srt and vtt are subtitle files. Recording only.
            summary (bool | Unset): Summarise the recording, where the provider offers audio intelligence. Recording only.
            entities (bool | Unset): Extract named entities from the recording. Recording only.
    """

    target: str | Unset = UNSET
    languages: list[str] | Unset = UNSET
    detect_language: bool | Unset = UNSET
    sample_rate: int | Unset = UNSET
    interim: bool | Unset = UNSET
    endpointing: Endpointing | Unset = UNSET
    silence_ms: int | Unset = UNSET
    utterance_end_ms: int | Unset = UNSET
    diarize: bool | Unset = UNSET
    max_speakers: int | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    format_: bool | Unset = UNSET
    redact: bool | Unset = UNSET
    events: bool | Unset = UNSET
    channels: int | Unset = UNSET
    words: bool | Unset = UNSET
    output: TranscriptFormat | Unset = UNSET
    summary: bool | Unset = UNSET
    entities: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        target = self.target

        languages: list[str] | Unset = UNSET
        if not isinstance(self.languages, Unset):
            languages = self.languages

        detect_language = self.detect_language

        sample_rate = self.sample_rate

        interim = self.interim

        endpointing: str | Unset = UNSET
        if not isinstance(self.endpointing, Unset):
            endpointing = self.endpointing.value

        silence_ms = self.silence_ms

        utterance_end_ms = self.utterance_end_ms

        diarize = self.diarize

        max_speakers = self.max_speakers

        keyterms: list[str] | Unset = UNSET
        if not isinstance(self.keyterms, Unset):
            keyterms = self.keyterms

        format_ = self.format_

        redact = self.redact

        events = self.events

        channels = self.channels

        words = self.words

        output: str | Unset = UNSET
        if not isinstance(self.output, Unset):
            output = self.output.value

        summary = self.summary

        entities = self.entities

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if target is not UNSET:
            field_dict["target"] = target
        if languages is not UNSET:
            field_dict["languages"] = languages
        if detect_language is not UNSET:
            field_dict["detect_language"] = detect_language
        if sample_rate is not UNSET:
            field_dict["sample_rate"] = sample_rate
        if interim is not UNSET:
            field_dict["interim"] = interim
        if endpointing is not UNSET:
            field_dict["endpointing"] = endpointing
        if silence_ms is not UNSET:
            field_dict["silence_ms"] = silence_ms
        if utterance_end_ms is not UNSET:
            field_dict["utterance_end_ms"] = utterance_end_ms
        if diarize is not UNSET:
            field_dict["diarize"] = diarize
        if max_speakers is not UNSET:
            field_dict["max_speakers"] = max_speakers
        if keyterms is not UNSET:
            field_dict["keyterms"] = keyterms
        if format_ is not UNSET:
            field_dict["format"] = format_
        if redact is not UNSET:
            field_dict["redact"] = redact
        if events is not UNSET:
            field_dict["events"] = events
        if channels is not UNSET:
            field_dict["channels"] = channels
        if words is not UNSET:
            field_dict["words"] = words
        if output is not UNSET:
            field_dict["output"] = output
        if summary is not UNSET:
            field_dict["summary"] = summary
        if entities is not UNSET:
            field_dict["entities"] = entities

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        target = d.pop("target", UNSET)

        languages = cast(list[str], d.pop("languages", UNSET))

        detect_language = d.pop("detect_language", UNSET)

        sample_rate = d.pop("sample_rate", UNSET)

        interim = d.pop("interim", UNSET)

        _endpointing = d.pop("endpointing", UNSET)
        endpointing: Endpointing | Unset
        if isinstance(_endpointing, Unset):
            endpointing = UNSET
        else:
            endpointing = Endpointing(_endpointing)

        silence_ms = d.pop("silence_ms", UNSET)

        utterance_end_ms = d.pop("utterance_end_ms", UNSET)

        diarize = d.pop("diarize", UNSET)

        max_speakers = d.pop("max_speakers", UNSET)

        keyterms = cast(list[str], d.pop("keyterms", UNSET))

        format_ = d.pop("format", UNSET)

        redact = d.pop("redact", UNSET)

        events = d.pop("events", UNSET)

        channels = d.pop("channels", UNSET)

        words = d.pop("words", UNSET)

        _output = d.pop("output", UNSET)
        output: TranscriptFormat | Unset
        if isinstance(_output, Unset):
            output = UNSET
        else:
            output = TranscriptFormat(_output)

        summary = d.pop("summary", UNSET)

        entities = d.pop("entities", UNSET)

        stt_options = cls(
            target=target,
            languages=languages,
            detect_language=detect_language,
            sample_rate=sample_rate,
            interim=interim,
            endpointing=endpointing,
            silence_ms=silence_ms,
            utterance_end_ms=utterance_end_ms,
            diarize=diarize,
            max_speakers=max_speakers,
            keyterms=keyterms,
            format_=format_,
            redact=redact,
            events=events,
            channels=channels,
            words=words,
            output=output,
            summary=summary,
            entities=entities,
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
