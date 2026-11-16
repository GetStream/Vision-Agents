from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.audit_entry import AuditEntry


T = TypeVar("T", bound="AgentChanges")


@_attrs_define
class AgentChanges:
    """What was changed about an agent since its directory was last synced: the edits a sync of that directory would write
    over.

        Attributes:
            items (list[AuditEntry]): The changes made since the last sync, newest first, at most 200. Empty when the agent
                has not been touched since.
            last_change (str | Unset): The newest change's id. Send it as base_change on a sync to say these have been seen,
                and that sync will not be refused for them.
            synced_at (datetime.datetime | Unset): When the directory was last synced onto this agent. Absent for an agent
                no directory has ever been synced onto.
    """

    items: list[AuditEntry]
    last_change: str | Unset = UNSET
    synced_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        last_change = self.last_change

        synced_at: str | Unset = UNSET
        if not isinstance(self.synced_at, Unset):
            synced_at = self.synced_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "items": items,
            }
        )
        if last_change is not UNSET:
            field_dict["last_change"] = last_change
        if synced_at is not UNSET:
            field_dict["synced_at"] = synced_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.audit_entry import AuditEntry

        d = dict(src_dict)
        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = AuditEntry.from_dict(items_item_data)

            items.append(items_item)

        last_change = d.pop("last_change", UNSET)

        _synced_at = d.pop("synced_at", UNSET)
        synced_at: datetime.datetime | Unset
        if isinstance(_synced_at, Unset):
            synced_at = UNSET
        else:
            synced_at = datetime.datetime.fromisoformat(_synced_at)

        agent_changes = cls(
            items=items,
            last_change=last_change,
            synced_at=synced_at,
        )

        agent_changes.additional_properties = d
        return agent_changes

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
