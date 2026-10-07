from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.cost_source import CostSource
    from ..models.input_parts import InputParts
    from ..models.model_tokens import ModelTokens


T = TypeVar("T", bound="CallTokens")


@_attrs_define
class CallTokens:
    """What a call's models read and wrote, summed over every request, with what their prompts were made of.

    Attributes:
        cached_input_tokens (int): The part of the prompts a provider served from its own cache.
        cost_micros (int): Millionths of a dollar, priced from the providers' configured rates.
        cost_sources (list[CostSource]): Where the cost came from, the costliest first. Sources that cost nothing are
            left out.
        input_parts (InputParts): What prompts were made of, in tokens. Estimated from each request and scaled to what
            the provider counted, so the parts sum to the input tokens and only the split between them is a guess. Requests
            recorded before the split was kept read zero throughout.
        input_tokens (int): Every prompt the models read, the cached part included.
        models (list[ModelTokens]): Each model the call used, the busiest first. One billed by audio or characters reads
            zero tokens.
        output_tokens (int): Everything the models generated, reasoning included.
        requests (int): How many calls to those models it took.
    """

    cached_input_tokens: int
    cost_micros: int
    cost_sources: list[CostSource]
    input_parts: InputParts
    input_tokens: int
    models: list[ModelTokens]
    output_tokens: int
    requests: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cached_input_tokens = self.cached_input_tokens

        cost_micros = self.cost_micros

        cost_sources = []
        for cost_sources_item_data in self.cost_sources:
            cost_sources_item = cost_sources_item_data.to_dict()
            cost_sources.append(cost_sources_item)

        input_parts = self.input_parts.to_dict()

        input_tokens = self.input_tokens

        models = []
        for models_item_data in self.models:
            models_item = models_item_data.to_dict()
            models.append(models_item)

        output_tokens = self.output_tokens

        requests = self.requests

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "cached_input_tokens": cached_input_tokens,
                "cost_micros": cost_micros,
                "cost_sources": cost_sources,
                "input_parts": input_parts,
                "input_tokens": input_tokens,
                "models": models,
                "output_tokens": output_tokens,
                "requests": requests,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.cost_source import CostSource
        from ..models.input_parts import InputParts
        from ..models.model_tokens import ModelTokens

        d = dict(src_dict)
        cached_input_tokens = d.pop("cached_input_tokens")

        cost_micros = d.pop("cost_micros")

        cost_sources = []
        _cost_sources = d.pop("cost_sources")
        for cost_sources_item_data in _cost_sources:
            cost_sources_item = CostSource.from_dict(cost_sources_item_data)

            cost_sources.append(cost_sources_item)

        input_parts = InputParts.from_dict(d.pop("input_parts"))

        input_tokens = d.pop("input_tokens")

        models = []
        _models = d.pop("models")
        for models_item_data in _models:
            models_item = ModelTokens.from_dict(models_item_data)

            models.append(models_item)

        output_tokens = d.pop("output_tokens")

        requests = d.pop("requests")

        call_tokens = cls(
            cached_input_tokens=cached_input_tokens,
            cost_micros=cost_micros,
            cost_sources=cost_sources,
            input_parts=input_parts,
            input_tokens=input_tokens,
            models=models,
            output_tokens=output_tokens,
            requests=requests,
        )

        call_tokens.additional_properties = d
        return call_tokens

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
