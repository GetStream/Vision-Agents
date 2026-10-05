from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.model_overwrites_thinking import ModelOverwritesThinking
from ..models.model_overwrites_verbosity import ModelOverwritesVerbosity
from ..types import UNSET, Unset

T = TypeVar("T", bound="ModelOverwrites")


@_attrs_define
class ModelOverwrites:
    """What to change about the models for one session, over whatever its agent config decided.
    It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides the
    config, and a caller reading a session back wants to see what they changed in one place rather than diffed against a
    config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because a caller able
    to rewrite those could make a session impersonate a different agent.

        Attributes:
            llm (str | Unset): A provider/model or a capability shortcut, in place of the config's.
            max_output_tokens (int | Unset): Caps the reply, reasoning included. Omitted leaves the provider's default.
            search (str | Unset):
            sts (str | Unset): A speech-to-speech target. Naming one here makes the session native even if the config did
                not, which means no transcriber, model or voice is opened.
            stt (str | Unset):
            temperature (float | Unset): How random the answer is. Omitted leaves the provider's own default, which is not
                the same as zero: zero is a real request for a deterministic model.
            thinking (ModelOverwritesThinking | Unset): How hard to reason before answering. It becomes the reasoning effort
                on the request, which is the vocabulary the providers that support one already speak, and means nothing to a
                model that does not reason.
            tts (str | Unset):
            verbosity (ModelOverwritesVerbosity | Unset): How much detail to give. Dropped for models that do not take it.
    """

    llm: str | Unset = UNSET
    max_output_tokens: int | Unset = UNSET
    search: str | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    temperature: float | Unset = UNSET
    thinking: ModelOverwritesThinking | Unset = UNSET
    tts: str | Unset = UNSET
    verbosity: ModelOverwritesVerbosity | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        llm = self.llm

        max_output_tokens = self.max_output_tokens

        search = self.search

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

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if llm is not UNSET:
            field_dict["llm"] = llm
        if max_output_tokens is not UNSET:
            field_dict["max_output_tokens"] = max_output_tokens
        if search is not UNSET:
            field_dict["search"] = search
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

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        llm = d.pop("llm", UNSET)

        max_output_tokens = d.pop("max_output_tokens", UNSET)

        search = d.pop("search", UNSET)

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        temperature = d.pop("temperature", UNSET)

        _thinking = d.pop("thinking", UNSET)
        thinking: ModelOverwritesThinking | Unset
        if isinstance(_thinking, Unset):
            thinking = UNSET
        else:
            thinking = ModelOverwritesThinking(_thinking)

        tts = d.pop("tts", UNSET)

        _verbosity = d.pop("verbosity", UNSET)
        verbosity: ModelOverwritesVerbosity | Unset
        if isinstance(_verbosity, Unset):
            verbosity = UNSET
        else:
            verbosity = ModelOverwritesVerbosity(_verbosity)

        model_overwrites = cls(
            llm=llm,
            max_output_tokens=max_output_tokens,
            search=search,
            sts=sts,
            stt=stt,
            temperature=temperature,
            thinking=thinking,
            tts=tts,
            verbosity=verbosity,
        )

        model_overwrites.additional_properties = d
        return model_overwrites

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
