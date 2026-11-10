from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.offered_tool_source import OfferedToolSource
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
        source (OfferedToolSource | Unset): Where the tool came from: session, declared by whoever opened the session;
            worker, hosted by a worker connected when the session opened; connector, opened by one of the agent's connector
            bindings; builtin, the router's own. A session's tools are fixed when it opens, so a worker that was not
            connected then leaves none of its tools here. Absent for a session kept before sources were recorded.
    """

    description: str
    name: str
    tokens: int
    parameters: OfferedToolParameters | Unset = UNSET
    source: OfferedToolSource | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        description = self.description

        name = self.name

        tokens = self.tokens

        parameters: dict[str, Any] | Unset = UNSET
        if not isinstance(self.parameters, Unset):
            parameters = self.parameters.to_dict()

        source: str | Unset = UNSET
        if not isinstance(self.source, Unset):
            source = self.source.value

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
        if source is not UNSET:
            field_dict["source"] = source

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.offered_tool_parameters import OfferedToolParameters

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

        _source = d.pop("source", UNSET)
        source: OfferedToolSource | Unset
        if isinstance(_source, Unset):
            source = UNSET
        else:
            source = OfferedToolSource._decode(_source)

        offered_tool = cls(
            description=description,
            name=name,
            tokens=tokens,
            parameters=parameters,
            source=source,
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
