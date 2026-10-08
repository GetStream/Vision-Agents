from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.input_parts import InputParts


T = TypeVar("T", bound="ModelTokens")


@_attrs_define
class ModelTokens:
    """What one model read and wrote over a call.

    Attributes:
        cached_input_tokens (int):
        cost_micros (int):
        input_parts (InputParts): What prompts were made of, in tokens. Estimated from each request and scaled to what
            the provider counted, so the parts sum to the input tokens and only the split between them is a guess. Requests
            recorded before the split was kept read zero throughout.
        input_tokens (int):
        modality (str): What the model does: llm for a language model, sts for speech-to-speech, and so on.
        model (str):
        output_tokens (int):
        provider (str):
        requests (int):
    """

    cached_input_tokens: int
    cost_micros: int
    input_parts: InputParts
    input_tokens: int
    modality: str
    model: str
    output_tokens: int
    provider: str
    requests: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cached_input_tokens = self.cached_input_tokens

        cost_micros = self.cost_micros

        input_parts = self.input_parts.to_dict()

        input_tokens = self.input_tokens

        modality = self.modality

        model = self.model

        output_tokens = self.output_tokens

        provider = self.provider

        requests = self.requests

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "cached_input_tokens": cached_input_tokens,
                "cost_micros": cost_micros,
                "input_parts": input_parts,
                "input_tokens": input_tokens,
                "modality": modality,
                "model": model,
                "output_tokens": output_tokens,
                "provider": provider,
                "requests": requests,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.input_parts import InputParts

        d = dict(src_dict)
        cached_input_tokens = d.pop("cached_input_tokens")

        cost_micros = d.pop("cost_micros")

        input_parts = InputParts.from_dict(d.pop("input_parts"))

        input_tokens = d.pop("input_tokens")

        modality = d.pop("modality")

        model = d.pop("model")

        output_tokens = d.pop("output_tokens")

        provider = d.pop("provider")

        requests = d.pop("requests")

        model_tokens = cls(
            cached_input_tokens=cached_input_tokens,
            cost_micros=cost_micros,
            input_parts=input_parts,
            input_tokens=input_tokens,
            modality=modality,
            model=model,
            output_tokens=output_tokens,
            provider=provider,
            requests=requests,
        )

        model_tokens.additional_properties = d
        return model_tokens

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
