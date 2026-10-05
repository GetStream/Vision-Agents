from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.update_session_request_thinking import UpdateSessionRequestThinking
from ..models.update_session_request_verbosity import UpdateSessionRequestVerbosity
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.update_session_request_custom import UpdateSessionRequestCustom


T = TypeVar("T", bound="UpdateSessionRequest")


@_attrs_define
class UpdateSessionRequest:
    """What to change about one session. A field left out is left as it is. Title, description and custom can change on a
    session that ended, and are all an end user's device may change; everything else needs the session running and a
    server-side caller.

        Attributes:
            custom (UpdateSessionRequestCustom | Unset): Replaces the caller's labels whole. An empty object clears them.
            description (str | Unset):
            instructions (str | Unset): What the agent is told to be, from the next turn.
            llm (str | Unset): The conversation model, a provider/model or a capability shortcut.
            max_output_tokens (int | Unset):
            sts (str | Unset): A speech-to-speech target, which makes the session native. Empty makes it a cascade again.
            stt (str | Unset):
            temperature (float | Unset):
            thinking (UpdateSessionRequestThinking | Unset):
            title (str | Unset):
            tts (str | Unset):
            verbosity (UpdateSessionRequestVerbosity | Unset):
            voice (str | Unset): The voice to speak in, in the provider's own terms. Empty returns to the provider's
                default.
    """

    custom: UpdateSessionRequestCustom | Unset = UNSET
    description: str | Unset = UNSET
    instructions: str | Unset = UNSET
    llm: str | Unset = UNSET
    max_output_tokens: int | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    temperature: float | Unset = UNSET
    thinking: UpdateSessionRequestThinking | Unset = UNSET
    title: str | Unset = UNSET
    tts: str | Unset = UNSET
    verbosity: UpdateSessionRequestVerbosity | Unset = UNSET
    voice: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        description = self.description

        instructions = self.instructions

        llm = self.llm

        max_output_tokens = self.max_output_tokens

        sts = self.sts

        stt = self.stt

        temperature = self.temperature

        thinking: str | Unset = UNSET
        if not isinstance(self.thinking, Unset):
            thinking = self.thinking.value

        title = self.title

        tts = self.tts

        verbosity: str | Unset = UNSET
        if not isinstance(self.verbosity, Unset):
            verbosity = self.verbosity.value

        voice = self.voice

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if custom is not UNSET:
            field_dict["custom"] = custom
        if description is not UNSET:
            field_dict["description"] = description
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
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
        if title is not UNSET:
            field_dict["title"] = title
        if tts is not UNSET:
            field_dict["tts"] = tts
        if verbosity is not UNSET:
            field_dict["verbosity"] = verbosity
        if voice is not UNSET:
            field_dict["voice"] = voice

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.update_session_request_custom import (
            UpdateSessionRequestCustom,
        )

        d = dict(src_dict)
        _custom = d.pop("custom", UNSET)
        custom: UpdateSessionRequestCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = UpdateSessionRequestCustom.from_dict(_custom)

        description = d.pop("description", UNSET)

        instructions = d.pop("instructions", UNSET)

        llm = d.pop("llm", UNSET)

        max_output_tokens = d.pop("max_output_tokens", UNSET)

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        temperature = d.pop("temperature", UNSET)

        _thinking = d.pop("thinking", UNSET)
        thinking: UpdateSessionRequestThinking | Unset
        if isinstance(_thinking, Unset):
            thinking = UNSET
        else:
            thinking = UpdateSessionRequestThinking(_thinking)

        title = d.pop("title", UNSET)

        tts = d.pop("tts", UNSET)

        _verbosity = d.pop("verbosity", UNSET)
        verbosity: UpdateSessionRequestVerbosity | Unset
        if isinstance(_verbosity, Unset):
            verbosity = UNSET
        else:
            verbosity = UpdateSessionRequestVerbosity(_verbosity)

        voice = d.pop("voice", UNSET)

        update_session_request = cls(
            custom=custom,
            description=description,
            instructions=instructions,
            llm=llm,
            max_output_tokens=max_output_tokens,
            sts=sts,
            stt=stt,
            temperature=temperature,
            thinking=thinking,
            title=title,
            tts=tts,
            verbosity=verbosity,
            voice=voice,
        )

        update_session_request.additional_properties = d
        return update_session_request

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
