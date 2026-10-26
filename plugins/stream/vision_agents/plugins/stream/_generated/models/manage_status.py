from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ManageStatus")


@_attrs_define
class ManageStatus:
    """Who a management call is recorded as made by: the actor and the name the call named, as the audit log keeps them.

    Attributes:
        actor (str): The X-Stream-Actor-Id the call named.
        name (str | Unset): The X-Stream-Actor-Name it named, decoded from percent-encoding. Absent when it named none.
    """

    actor: str
    name: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        actor = self.actor

        name = self.name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "actor": actor,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        actor = d.pop("actor")

        name = d.pop("name", UNSET)

        manage_status = cls(
            actor=actor,
            name=name,
        )

        manage_status.additional_properties = d
        return manage_status

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
