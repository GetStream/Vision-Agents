from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="CallUsage")


@_attrs_define
class CallUsage:
    """What the call spent, summed over every request it made. Counted once the call is over, so it is absent while one is
    still running. Requests that failed are included: a model that read the prompt and then fell over is still billed
    for it.

        Attributes:
            input_tokens (int): Every prompt the models read, the cached part included.
            cached_input_tokens (int): The part of those prompts a provider served from its own cache.
            output_tokens (int): Everything the models generated, reasoning included.
            cost_micros (int): Millionths of a dollar, priced from the providers' configured rates.
            requests (int): How many calls to a model it took, transcription and speech included.
    """

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_micros: int
    requests: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        input_tokens = self.input_tokens

        cached_input_tokens = self.cached_input_tokens

        output_tokens = self.output_tokens

        cost_micros = self.cost_micros

        requests = self.requests

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
                "cost_micros": cost_micros,
                "requests": requests,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        input_tokens = d.pop("input_tokens")

        cached_input_tokens = d.pop("cached_input_tokens")

        output_tokens = d.pop("output_tokens")

        cost_micros = d.pop("cost_micros")

        requests = d.pop("requests")

        call_usage = cls(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            cost_micros=cost_micros,
            requests=requests,
        )

        call_usage.additional_properties = d
        return call_usage

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
