from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_tool_parameters import SessionToolParameters


T = TypeVar("T", bound="SessionTool")


@_attrs_define
class SessionTool:
    """One of the caller's own functions. The model is offered it by name and description; running it is the caller's
    business, over the events socket.

        Attributes:
            name (str):
            description (str): What the model is told the tool does, which is the whole of how it decides when to reach for
                one.
            parameters (SessionToolParameters | Unset): A JSON Schema object describing the arguments.
    """

    name: str
    description: str
    parameters: SessionToolParameters | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        description = self.description

        parameters: dict[str, Any] | Unset = UNSET
        if not isinstance(self.parameters, Unset):
            parameters = self.parameters.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "description": description,
            }
        )
        if parameters is not UNSET:
            field_dict["parameters"] = parameters

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.session_tool_parameters import SessionToolParameters

        d = dict(src_dict)
        name = d.pop("name")

        description = d.pop("description")

        _parameters = d.pop("parameters", UNSET)
        parameters: SessionToolParameters | Unset
        if isinstance(_parameters, Unset):
            parameters = UNSET
        else:
            parameters = SessionToolParameters.from_dict(_parameters)

        session_tool = cls(
            name=name,
            description=description,
            parameters=parameters,
        )

        session_tool.additional_properties = d
        return session_tool

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
