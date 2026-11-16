from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="CostSource")


@_attrs_define
class CostSource:
    """One place a call's cost came from.

    Attributes:
        cost_micros (int): Millionths of a dollar. A part of the prompt is given its share of what the prompt cost, by
            tokens.
        source (str): For a model billed by tokens, the part of its prompt (instructions, messages, tool_definitions,
            tool_use, images or video), output for what it wrote, or input for prompt tokens recorded without a breakdown.
            For any other model, its modality: stt, tts, search and the rest.
    """

    cost_micros: int
    source: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cost_micros = self.cost_micros

        source = self.source

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "cost_micros": cost_micros,
                "source": source,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        cost_micros = d.pop("cost_micros")

        source = d.pop("source")

        cost_source = cls(
            cost_micros=cost_micros,
            source=source,
        )

        cost_source.additional_properties = d
        return cost_source

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
