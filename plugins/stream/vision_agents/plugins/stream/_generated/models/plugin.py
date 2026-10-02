from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="Plugin")


@_attrs_define
class Plugin:
    """One hosted MCP server from the built-in catalog.

    Attributes:
        category (str):
        description (str):
        id (str):
        name (str):
        instance_hint (str | Unset):
        instance_required (bool | Unset):
    """

    category: str
    description: str
    id: str
    name: str
    instance_hint: str | Unset = UNSET
    instance_required: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        category = self.category

        description = self.description

        id = self.id

        name = self.name

        instance_hint = self.instance_hint

        instance_required = self.instance_required

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "category": category,
                "description": description,
                "id": id,
                "name": name,
            }
        )
        if instance_hint is not UNSET:
            field_dict["instance_hint"] = instance_hint
        if instance_required is not UNSET:
            field_dict["instance_required"] = instance_required

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        category = d.pop("category")

        description = d.pop("description")

        id = d.pop("id")

        name = d.pop("name")

        instance_hint = d.pop("instance_hint", UNSET)

        instance_required = d.pop("instance_required", UNSET)

        plugin = cls(
            category=category,
            description=description,
            id=id,
            name=name,
            instance_hint=instance_hint,
            instance_required=instance_required,
        )

        plugin.additional_properties = d
        return plugin

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
