from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.offered_tool_parameters import OfferedToolParameters


T = TypeVar("T", bound="OfferedTool")


@_attrs_define
class OfferedTool:
    """One tool as the model sees it.

    Attributes:
        description (str): What the model is told the tool does.
        name (str): How the model asks for it.
        tokens (int): Roughly what offering it costs on every request, in tokens: its name, description and schema at
            four characters a token.
        parameters (OfferedToolParameters | Unset): The JSON Schema of its arguments.
    """

    description: str
    name: str
    tokens: int
    parameters: OfferedToolParameters | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        description = self.description

        name = self.name

        tokens = self.tokens

        parameters: dict[str, Any] | Unset = UNSET
        if not isinstance(self.parameters, Unset):
            parameters = self.parameters.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "description": description,
                "name": name,
                "tokens": tokens,
            }
        )
        if parameters is not UNSET:
            field_dict["parameters"] = parameters

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.offered_tool_parameters import (
            OfferedToolParameters,
        )

        d = dict(src_dict)
        description = d.pop("description")

        name = d.pop("name")

        tokens = d.pop("tokens")

        _parameters = d.pop("parameters", UNSET)
        parameters: OfferedToolParameters | Unset
        if isinstance(_parameters, Unset):
            parameters = UNSET
        else:
            parameters = OfferedToolParameters.from_dict(_parameters)

        offered_tool = cls(
            description=description,
            name=name,
            tokens=tokens,
            parameters=parameters,
        )

        offered_tool.additional_properties = d
        return offered_tool

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
