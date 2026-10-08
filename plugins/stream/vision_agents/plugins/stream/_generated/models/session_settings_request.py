from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.session_settings_request_thinking import SessionSettingsRequestThinking
from ..models.session_settings_request_verbosity import SessionSettingsRequestVerbosity
from ..types import UNSET, Unset

T = TypeVar("T", bound="SessionSettingsRequest")


@_attrs_define
class SessionSettingsRequest:
    """What to change about one running session's models. A field left out is left as it is. The same safe knobs as
    ModelOverwrites, plus the voice.

        Attributes:
            llm (str | Unset): The conversation model, a provider/model or a capability shortcut.
            max_output_tokens (int | Unset):
            sts (str | Unset): A speech-to-speech target, which makes the session native. Empty makes it a cascade again.
            stt (str | Unset):
            temperature (float | Unset):
            thinking (SessionSettingsRequestThinking | Unset):
            tts (str | Unset):
            verbosity (SessionSettingsRequestVerbosity | Unset):
            voice (str | Unset): The voice to speak in, in the provider's own terms. Empty returns to the provider's
                default.
    """

    llm: str | Unset = UNSET
    max_output_tokens: int | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    temperature: float | Unset = UNSET
    thinking: SessionSettingsRequestThinking | Unset = UNSET
    tts: str | Unset = UNSET
    verbosity: SessionSettingsRequestVerbosity | Unset = UNSET
    voice: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        llm = self.llm

        max_output_tokens = self.max_output_tokens

        sts = self.sts

        stt = self.stt

        temperature = self.temperature

        thinking: str | Unset = UNSET
        if not isinstance(self.thinking, Unset):
            thinking = self.thinking.value

        tts = self.tts

        verbosity: str | Unset = UNSET
        if not isinstance(self.verbosity, Unset):
            verbosity = self.verbosity.value

        voice = self.voice

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if llm is not UNSET:
            field_dict["llm"] = llm
        if max_output_tokens is not UNSET:
            field_dict["max_output_tokens"] = max_output_tokens
        if sts is not UNSET:
            field_dict["sts"] = sts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if temperature is not UNSET:
            field_dict["temperature"] = temperature
        if thinking is not UNSET:
            field_dict["thinking"] = thinking
        if tts is not UNSET:
            field_dict["tts"] = tts
        if verbosity is not UNSET:
            field_dict["verbosity"] = verbosity
        if voice is not UNSET:
            field_dict["voice"] = voice

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        llm = d.pop("llm", UNSET)

        max_output_tokens = d.pop("max_output_tokens", UNSET)

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        temperature = d.pop("temperature", UNSET)

        _thinking = d.pop("thinking", UNSET)
        thinking: SessionSettingsRequestThinking | Unset
        if isinstance(_thinking, Unset):
            thinking = UNSET
        else:
            thinking = SessionSettingsRequestThinking(_thinking)

        tts = d.pop("tts", UNSET)

        _verbosity = d.pop("verbosity", UNSET)
        verbosity: SessionSettingsRequestVerbosity | Unset
        if isinstance(_verbosity, Unset):
            verbosity = UNSET
        else:
            verbosity = SessionSettingsRequestVerbosity(_verbosity)

        voice = d.pop("voice", UNSET)

        session_settings_request = cls(
            llm=llm,
            max_output_tokens=max_output_tokens,
            sts=sts,
            stt=stt,
            temperature=temperature,
            thinking=thinking,
            tts=tts,
            verbosity=verbosity,
            voice=voice,
        )

        session_settings_request.additional_properties = d
        return session_settings_request

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
