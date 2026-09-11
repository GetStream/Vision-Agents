from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.sts_options_turn_detection import StsOptionsTurnDetection
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.data_policy import DataPolicy
    from ..models.sts_options_overwrites import StsOptionsOverwrites


T = TypeVar("T", bound="StsOptions")


@_attrs_define
class StsOptions:
    """How this config holds a conversation with one native audio model, in place of a transcriber, a text model and a
    voice. What every such model takes is a field here; what only some take is a term, and a request naming a term is
    routed to a model that declared it or refused, never served by one that ignores it.

        Attributes:
            target (str | Unset): A provider/model or a capability shortcut. Example: sts-fast.
            providers (list[str] | Unset): A priority list of where to try, in the order given, which wins over target when
                it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it
                stands.
                 Example: ['openai', 'sts-fast'].
            instructions (str | Unset): The system prompt the model converses under.
            voice (str | Unset): The vendor's own name for a voice, such as marin at OpenAI or Kore at Google. None of these
                models takes one of your own voices, so the name is passed on as given rather than looked up.
                 Example: marin.
            languages (list[str] | Unset):
            turn_detection (StsOptionsTurnDetection | Unset): What decides the caller has finished: a silence timer, a model
                reading the words, or nothing, which leaves the turns to the caller. Omitting it leaves the vendor's default.
                Only some models read the words, so semantic is a term.
            silence_ms (int | Unset): How long a pause ends the turn, for a silence timer.
            prefix_padding_ms (int | Unset): How much audio before the detected speech is kept, for a silence timer.
            interrupt_response (bool | Unset): Whether the model cuts its own reply off when it hears the caller. Omitting
                it leaves the vendor's default; false is for a speaker close enough to the microphone that the model would
                otherwise interrupt itself.
            input_transcript (bool | Unset): Ask the model to write down what it heard.
            output_transcript (bool | Unset): Ask the model to write down what it said.
            tools (bool | Unset): The session will hand the model functions to call.
            text (bool | Unset): The session will inject typed turns.
            images (bool | Unset): The session will send the model frames, so it is routed only to a model that sees, the
                way vlm routes a text model.
            data_policy (DataPolicy | Unset): What a caller requires of what happens to what they send: the audio they had
                transcribed, or the text they had spoken and the voice speaking it. This is a requirement rather than a
                description: a request naming one is only routed to a model whose declared handling meets it, and if none does
                the request is refused rather than sent somewhere that does not.
            overwrites (StsOptionsOverwrites | Unset): Settings for one provider that this vocabulary has no word for, keyed
                by provider name, for example {"openai": {"eagerness": "high"}}. The provider named parses its own block and
                refuses a field it does not have, so an overwrite is either sent or reported rather than accepted and dropped.
                 Example: {'openai': {'eagerness': 'high'}}.
    """

    target: str | Unset = UNSET
    providers: list[str] | Unset = UNSET
    instructions: str | Unset = UNSET
    voice: str | Unset = UNSET
    languages: list[str] | Unset = UNSET
    turn_detection: StsOptionsTurnDetection | Unset = UNSET
    silence_ms: int | Unset = UNSET
    prefix_padding_ms: int | Unset = UNSET
    interrupt_response: bool | Unset = UNSET
    input_transcript: bool | Unset = UNSET
    output_transcript: bool | Unset = UNSET
    tools: bool | Unset = UNSET
    text: bool | Unset = UNSET
    images: bool | Unset = UNSET
    data_policy: DataPolicy | Unset = UNSET
    overwrites: StsOptionsOverwrites | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        target = self.target

        providers: list[str] | Unset = UNSET
        if not isinstance(self.providers, Unset):
            providers = self.providers

        instructions = self.instructions

        voice = self.voice

        languages: list[str] | Unset = UNSET
        if not isinstance(self.languages, Unset):
            languages = self.languages

        turn_detection: str | Unset = UNSET
        if not isinstance(self.turn_detection, Unset):
            turn_detection = self.turn_detection.value

        silence_ms = self.silence_ms

        prefix_padding_ms = self.prefix_padding_ms

        interrupt_response = self.interrupt_response

        input_transcript = self.input_transcript

        output_transcript = self.output_transcript

        tools = self.tools

        text = self.text

        images = self.images

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
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if voice is not UNSET:
            field_dict["voice"] = voice
        if languages is not UNSET:
            field_dict["languages"] = languages
        if turn_detection is not UNSET:
            field_dict["turn_detection"] = turn_detection
        if silence_ms is not UNSET:
            field_dict["silence_ms"] = silence_ms
        if prefix_padding_ms is not UNSET:
            field_dict["prefix_padding_ms"] = prefix_padding_ms
        if interrupt_response is not UNSET:
            field_dict["interrupt_response"] = interrupt_response
        if input_transcript is not UNSET:
            field_dict["input_transcript"] = input_transcript
        if output_transcript is not UNSET:
            field_dict["output_transcript"] = output_transcript
        if tools is not UNSET:
            field_dict["tools"] = tools
        if text is not UNSET:
            field_dict["text"] = text
        if images is not UNSET:
            field_dict["images"] = images
        if data_policy is not UNSET:
            field_dict["data_policy"] = data_policy
        if overwrites is not UNSET:
            field_dict["overwrites"] = overwrites

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.data_policy import DataPolicy
        from ..models.sts_options_overwrites import (
            StsOptionsOverwrites,
        )

        d = dict(src_dict)
        target = d.pop("target", UNSET)

        providers = cast(list[str], d.pop("providers", UNSET))

        instructions = d.pop("instructions", UNSET)

        voice = d.pop("voice", UNSET)

        languages = cast(list[str], d.pop("languages", UNSET))

        _turn_detection = d.pop("turn_detection", UNSET)
        turn_detection: StsOptionsTurnDetection | Unset
        if isinstance(_turn_detection, Unset):
            turn_detection = UNSET
        else:
            turn_detection = StsOptionsTurnDetection(_turn_detection)

        silence_ms = d.pop("silence_ms", UNSET)

        prefix_padding_ms = d.pop("prefix_padding_ms", UNSET)

        interrupt_response = d.pop("interrupt_response", UNSET)

        input_transcript = d.pop("input_transcript", UNSET)

        output_transcript = d.pop("output_transcript", UNSET)

        tools = d.pop("tools", UNSET)

        text = d.pop("text", UNSET)

        images = d.pop("images", UNSET)

        _data_policy = d.pop("data_policy", UNSET)
        data_policy: DataPolicy | Unset
        if isinstance(_data_policy, Unset):
            data_policy = UNSET
        else:
            data_policy = DataPolicy.from_dict(_data_policy)

        _overwrites = d.pop("overwrites", UNSET)
        overwrites: StsOptionsOverwrites | Unset
        if isinstance(_overwrites, Unset):
            overwrites = UNSET
        else:
            overwrites = StsOptionsOverwrites.from_dict(_overwrites)

        sts_options = cls(
            target=target,
            providers=providers,
            instructions=instructions,
            voice=voice,
            languages=languages,
            turn_detection=turn_detection,
            silence_ms=silence_ms,
            prefix_padding_ms=prefix_padding_ms,
            interrupt_response=interrupt_response,
            input_transcript=input_transcript,
            output_transcript=output_transcript,
            tools=tools,
            text=text,
            images=images,
            data_policy=data_policy,
            overwrites=overwrites,
        )

        sts_options.additional_properties = d
        return sts_options

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
