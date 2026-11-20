from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.audit_resource_type import AuditResourceType
from ..types import UNSET, Unset

T = TypeVar("T", bound="SyncConflict")


@_attrs_define
class SyncConflict:
    """One thing a sync would write over that somebody changed since the last sync.

    Attributes:
        name (str): Which one: the field of an agent_config, such as llm or instructions; a skill's name; a knowledge
            document's source; a knowledge url's url.
        resource_type (AuditResourceType): What the change was made to. All of them are configuration: what an agent
            does while it runs is traffic, and is read from the sessions and the logs instead. price_list is the
            deployment's own, changed through the management API, and is never listed for an app. end_user is one of the
            app's end users whose id was forgotten in the usage records, named by the id that replaced it. limit is one of
            the organization's spend limits, named by its id, a reset of one, named reset:<id>, or a grant, named
            grant:<id>; a change staff made through the management API is kept under the organization's id. provider_key is
            a key the app keeps for its own account at a vendor, named <provider>/<region>; its changes show the key's last
            four characters at most, never the key.
        removed (bool | Unset): True for a knowledge document the directory no longer has, which the sync would delete.
    """

    name: str
    resource_type: AuditResourceType
    removed: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        resource_type = self.resource_type.value

        removed = self.removed

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "resource_type": resource_type,
            }
        )
        if removed is not UNSET:
            field_dict["removed"] = removed

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        resource_type = AuditResourceType._decode(d.pop("resource_type"))

        removed = d.pop("removed", UNSET)

        sync_conflict = cls(
            name=name,
            resource_type=resource_type,
            removed=removed,
        )

        sync_conflict.additional_properties = d
        return sync_conflict

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
