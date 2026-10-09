from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.invocation_argument_type import InvocationArgumentType
from ..types import UNSET, Unset

T = TypeVar("T", bound="InvocationArgument")


@_attrs_define
class InvocationArgument:
    """One argument a connector tool call was asked with: its name, its JSON type and, for a string or an array, its
    length, so an empty string shows as length 0. Never its value.

        Attributes:
            name (str):
            type_ (InvocationArgumentType): The argument's JSON type.
            length (int | Unset): A string's characters (Unicode code points) or an array's elements. Absent for any other
                type.
    """

    name: str
    type_: InvocationArgumentType
    length: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        type_ = self.type_.value

        length = self.length

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "type": type_,
            }
        )
        if length is not UNSET:
            field_dict["length"] = length

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        type_ = InvocationArgumentType(d.pop("type"))

        length = d.pop("length", UNSET)

        invocation_argument = cls(
            name=name,
            type_=type_,
            length=length,
        )

        invocation_argument.additional_properties = d
        return invocation_argument

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
