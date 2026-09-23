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
            stt (str | Unset):
            tts (str | Unset):
            sts (str | Unset): A speech-to-speech target, which makes the session native. Empty makes it a cascade again.
            subagent (str | Unset):
            voice (str | Unset): The voice to speak in, in the provider's own terms. Empty returns to the provider's
                default.
            thinking (SessionSettingsRequestThinking | Unset):
            temperature (float | Unset):
            max_output_tokens (int | Unset):
            verbosity (SessionSettingsRequestVerbosity | Unset):
    """

    llm: str | Unset = UNSET
    stt: str | Unset = UNSET
    tts: str | Unset = UNSET
    sts: str | Unset = UNSET
    subagent: str | Unset = UNSET
    voice: str | Unset = UNSET
    thinking: SessionSettingsRequestThinking | Unset = UNSET
    temperature: float | Unset = UNSET
    max_output_tokens: int | Unset = UNSET
    verbosity: SessionSettingsRequestVerbosity | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        llm = self.llm

        stt = self.stt

        tts = self.tts

        sts = self.sts

        subagent = self.subagent

        voice = self.voice

        thinking: str | Unset = UNSET
        if not isinstance(self.thinking, Unset):
            thinking = self.thinking.value

        temperature = self.temperature

        max_output_tokens = self.max_output_tokens

        verbosity: str | Unset = UNSET
        if not isinstance(self.verbosity, Unset):
            verbosity = self.verbosity.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if llm is not UNSET:
            field_dict["llm"] = llm
        if stt is not UNSET:
            field_dict["stt"] = stt
        if tts is not UNSET:
            field_dict["tts"] = tts
        if sts is not UNSET:
            field_dict["sts"] = sts
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if voice is not UNSET:
            field_dict["voice"] = voice
        if thinking is not UNSET:
            field_dict["thinking"] = thinking
        if temperature is not UNSET:
            field_dict["temperature"] = temperature
        if max_output_tokens is not UNSET:
            field_dict["max_output_tokens"] = max_output_tokens
        if verbosity is not UNSET:
            field_dict["verbosity"] = verbosity

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        llm = d.pop("llm", UNSET)

        stt = d.pop("stt", UNSET)

        tts = d.pop("tts", UNSET)

        sts = d.pop("sts", UNSET)

        subagent = d.pop("subagent", UNSET)

        voice = d.pop("voice", UNSET)

        _thinking = d.pop("thinking", UNSET)
        thinking: SessionSettingsRequestThinking | Unset
        if isinstance(_thinking, Unset):
            thinking = UNSET
        else:
            thinking = SessionSettingsRequestThinking(_thinking)

        temperature = d.pop("temperature", UNSET)

        max_output_tokens = d.pop("max_output_tokens", UNSET)

        _verbosity = d.pop("verbosity", UNSET)
        verbosity: SessionSettingsRequestVerbosity | Unset
        if isinstance(_verbosity, Unset):
            verbosity = UNSET
        else:
            verbosity = SessionSettingsRequestVerbosity(_verbosity)

        session_settings_request = cls(
            llm=llm,
            stt=stt,
            tts=tts,
            sts=sts,
            subagent=subagent,
            voice=voice,
            thinking=thinking,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            verbosity=verbosity,
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
