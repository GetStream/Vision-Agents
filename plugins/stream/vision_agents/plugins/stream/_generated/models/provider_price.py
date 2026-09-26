from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ProviderPrice")


@_attrs_define
class ProviderPrice:
    """What this deployment is billed for the model, in US dollars. A rate is absent when the model is not billed by that
    unit.

        Attributes:
            per_million_input_tokens (float | Unset):  Example: 0.75.
            per_million_output_tokens (float | Unset):  Example: 3.75.
    """

    per_million_input_tokens: float | Unset = UNSET
    per_million_output_tokens: float | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        per_million_input_tokens = self.per_million_input_tokens

        per_million_output_tokens = self.per_million_output_tokens

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if per_million_input_tokens is not UNSET:
            field_dict["per_million_input_tokens"] = per_million_input_tokens
        if per_million_output_tokens is not UNSET:
            field_dict["per_million_output_tokens"] = per_million_output_tokens

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        per_million_input_tokens = d.pop("per_million_input_tokens", UNSET)

        per_million_output_tokens = d.pop("per_million_output_tokens", UNSET)

        provider_price = cls(
            per_million_input_tokens=per_million_input_tokens,
            per_million_output_tokens=per_million_output_tokens,
        )

        provider_price.additional_properties = d
        return provider_price

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
