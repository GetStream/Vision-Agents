from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorInput")


@_attrs_define
class ConnectorInput:
    """
    Attributes:
        name (str):
        default (str | Unset): Used when the connection gives no value. An input without one is required.
        enum (list[str] | None | Unset): The values it may take. Absent when a pattern decides.
        pattern (str | Unset): A regular expression the whole value must match.
    """

    name: str
    default: str | Unset = UNSET
    enum: list[str] | None | Unset = UNSET
    pattern: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        default = self.default

        enum: list[str] | None | Unset
        if isinstance(self.enum, Unset):
            enum = UNSET
        elif isinstance(self.enum, list):
            enum = self.enum

        else:
            enum = self.enum

        pattern = self.pattern

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
            }
        )
        if default is not UNSET:
            field_dict["default"] = default
        if enum is not UNSET:
            field_dict["enum"] = enum
        if pattern is not UNSET:
            field_dict["pattern"] = pattern

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        default = d.pop("default", UNSET)

        def _parse_enum(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                enum_type_0 = cast(list[str], data)

                return enum_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        enum = _parse_enum(d.pop("enum", UNSET))

        pattern = d.pop("pattern", UNSET)

        connector_input = cls(
            name=name,
            default=default,
            enum=enum,
            pattern=pattern,
        )

        connector_input.additional_properties = d
        return connector_input

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
