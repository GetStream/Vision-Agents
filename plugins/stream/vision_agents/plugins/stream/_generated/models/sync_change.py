from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.sync_change_action import SyncChangeAction
from ..models.sync_change_resource_type import SyncChangeResourceType

T = TypeVar("T", bound="SyncChange")


@_attrs_define
class SyncChange:
    """One skill, knowledge document, knowledge url or simulation a sync creates, updates or deletes.

    Attributes:
        action (SyncChangeAction):
        name (str): Which one: a skill's name, a knowledge document's source, a knowledge url's url, a simulation's
            name.
        resource_type (SyncChangeResourceType): What it is, named as GET /v1/agents/configs/{id}/changes names a skill,
            a knowledge document and a knowledge url, and simulation, which that record does not keep.
    """

    action: SyncChangeAction
    name: str
    resource_type: SyncChangeResourceType
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        action = self.action.value

        name = self.name

        resource_type = self.resource_type.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "action": action,
                "name": name,
                "resource_type": resource_type,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        action = SyncChangeAction._decode(d.pop("action"))

        name = d.pop("name")

        resource_type = SyncChangeResourceType._decode(d.pop("resource_type"))

        sync_change = cls(
            action=action,
            name=name,
            resource_type=resource_type,
        )

        sync_change.additional_properties = d
        return sync_change

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
