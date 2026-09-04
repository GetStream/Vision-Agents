from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.llm_options_format import LlmOptionsFormat
from ..models.llm_options_reasoning_effort import LlmOptionsReasoningEffort
from ..models.llm_options_verbosity import LlmOptionsVerbosity
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.llm_options_metadata import LlmOptionsMetadata


T = TypeVar("T", bound="LlmOptions")


@_attrs_define
class LlmOptions:
    """How this config answers. The names are the response parameters the router already speaks rather than a second
    vocabulary for the same things.

        Attributes:
            target (str | Unset): A provider/model or a capability shortcut. Example: llm-fast.
            instructions (str | Unset): What the model answers under, when a request does not say.
            max_output_tokens (int | Unset):
            temperature (float | Unset):
            reasoning_effort (LlmOptionsReasoningEffort | Unset): How long the model may think before answering, on the
                models that think.
            format_ (LlmOptionsFormat | Unset): Whether the answer is prose or a JSON object.
            verbosity (LlmOptionsVerbosity | Unset):
            tool_choice (str | Unset): auto, none, required, or the name of a tool the model must call. Which tools exist is
                per-request, since they change with the turn.
            store (bool | Unset): Keep the response on the provider so a later one can continue from it.
            prompt_cache_key (str | Unset): What a cached prompt prefix is keyed by. Requests sharing a key and a prefix are
                read from the cache rather than charged in full.
            metadata (LlmOptionsMetadata | Unset): Passed to the provider untouched, for the providers that store it.
    """

    target: str | Unset = UNSET
    instructions: str | Unset = UNSET
    max_output_tokens: int | Unset = UNSET
    temperature: float | Unset = UNSET
    reasoning_effort: LlmOptionsReasoningEffort | Unset = UNSET
    format_: LlmOptionsFormat | Unset = UNSET
    verbosity: LlmOptionsVerbosity | Unset = UNSET
    tool_choice: str | Unset = UNSET
    store: bool | Unset = UNSET
    prompt_cache_key: str | Unset = UNSET
    metadata: LlmOptionsMetadata | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        target = self.target

        instructions = self.instructions

        max_output_tokens = self.max_output_tokens

        temperature = self.temperature

        reasoning_effort: str | Unset = UNSET
        if not isinstance(self.reasoning_effort, Unset):
            reasoning_effort = self.reasoning_effort.value

        format_: str | Unset = UNSET
        if not isinstance(self.format_, Unset):
            format_ = self.format_.value

        verbosity: str | Unset = UNSET
        if not isinstance(self.verbosity, Unset):
            verbosity = self.verbosity.value

        tool_choice = self.tool_choice

        store = self.store

        prompt_cache_key = self.prompt_cache_key

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if target is not UNSET:
            field_dict["target"] = target
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if max_output_tokens is not UNSET:
            field_dict["max_output_tokens"] = max_output_tokens
        if temperature is not UNSET:
            field_dict["temperature"] = temperature
        if reasoning_effort is not UNSET:
            field_dict["reasoning_effort"] = reasoning_effort
        if format_ is not UNSET:
            field_dict["format"] = format_
        if verbosity is not UNSET:
            field_dict["verbosity"] = verbosity
        if tool_choice is not UNSET:
            field_dict["tool_choice"] = tool_choice
        if store is not UNSET:
            field_dict["store"] = store
        if prompt_cache_key is not UNSET:
            field_dict["prompt_cache_key"] = prompt_cache_key
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.llm_options_metadata import LlmOptionsMetadata

        d = dict(src_dict)
        target = d.pop("target", UNSET)

        instructions = d.pop("instructions", UNSET)

        max_output_tokens = d.pop("max_output_tokens", UNSET)

        temperature = d.pop("temperature", UNSET)

        _reasoning_effort = d.pop("reasoning_effort", UNSET)
        reasoning_effort: LlmOptionsReasoningEffort | Unset
        if isinstance(_reasoning_effort, Unset):
            reasoning_effort = UNSET
        else:
            reasoning_effort = LlmOptionsReasoningEffort(_reasoning_effort)

        _format_ = d.pop("format", UNSET)
        format_: LlmOptionsFormat | Unset
        if isinstance(_format_, Unset):
            format_ = UNSET
        else:
            format_ = LlmOptionsFormat(_format_)

        _verbosity = d.pop("verbosity", UNSET)
        verbosity: LlmOptionsVerbosity | Unset
        if isinstance(_verbosity, Unset):
            verbosity = UNSET
        else:
            verbosity = LlmOptionsVerbosity(_verbosity)

        tool_choice = d.pop("tool_choice", UNSET)

        store = d.pop("store", UNSET)

        prompt_cache_key = d.pop("prompt_cache_key", UNSET)

        _metadata = d.pop("metadata", UNSET)
        metadata: LlmOptionsMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = LlmOptionsMetadata.from_dict(_metadata)

        llm_options = cls(
            target=target,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            format_=format_,
            verbosity=verbosity,
            tool_choice=tool_choice,
            store=store,
            prompt_cache_key=prompt_cache_key,
            metadata=metadata,
        )

        llm_options.additional_properties = d
        return llm_options

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
