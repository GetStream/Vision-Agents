from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ClassifyUsage")


@_attrs_define
class ClassifyUsage:
    """What the request read and wrote. The state's tokens are counted once however many questions shared them.

    Attributes:
        input_tokens (int):
        output_tokens (int):
        cost_micros (int | Unset): What the request cost, in millionths of a dollar, as X-Stream-Cost-Micros says. Only
            a server-side caller is told.
    """

    input_tokens: int
    output_tokens: int
    cost_micros: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        input_tokens = self.input_tokens

        output_tokens = self.output_tokens

        cost_micros = self.cost_micros

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        if cost_micros is not UNSET:
            field_dict["cost_micros"] = cost_micros

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        input_tokens = d.pop("input_tokens")

        output_tokens = d.pop("output_tokens")

        cost_micros = d.pop("cost_micros", UNSET)

        classify_usage = cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_micros=cost_micros,
        )

        classify_usage.additional_properties = d
        return classify_usage

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
