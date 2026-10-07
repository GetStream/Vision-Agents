from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.audit_action import AuditAction
from ..models.audit_resource_type import AuditResourceType
from ..models.audit_source import AuditSource
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.audit_change import AuditChange


T = TypeVar("T", bound="AuditEntry")


@_attrs_define
class AuditEntry:
    """One change somebody made to the app's configuration: what changed, who changed it, which client they used, and the
    before and after of every field that moved. Only configuration is recorded -- agents, skills, knowledge, routers,
    plugins and policies -- never what an agent did while it ran.

        Attributes:
            action (AuditAction): created, updated or deleted for a change somebody made one at a time. synced is a whole
                agent directory written at once by POST /v1/agents/sync, and is what a later sync measures the edits made since
                against.
            changes (list[AuditChange]): Every field that moved. A write that moved nothing is not recorded at all, so this
                is empty only on a synced entry, which marks the moment a directory and an agent agreed whether or not anything
                moved.
            created_at (datetime.datetime):
            id (str):
            resource_id (str): The resource, which may since have been deleted.
            resource_type (AuditResourceType): What the change was made to. All of them are configuration: what an agent
                does while it runs is traffic, and is read from the sessions and the logs instead.
            source (AuditSource): Which client made it, from the X-Stream-Client header. api is a caller that named no
                client: it reached the API directly, which is all that can be said about it.
            actor_id (str | Unset): Who made it, as their client named them. Absent for a change nobody signed, such as a
                process syncing on startup.
            actor_name (str | Unset): Their name, as their client named them. Never an email address: the router keeps none.
            agent_id (str | Unset): The agent the change was to or under. Absent for a resource that belongs to no agent,
                such as a router.
            request_id (str | Unset): The X-Request-Id of the request that made it.
            resource_name (str | Unset): What it was called when it changed, for a resource that has a name.
    """

    action: AuditAction
    changes: list[AuditChange]
    created_at: datetime.datetime
    id: str
    resource_id: str
    resource_type: AuditResourceType
    source: AuditSource
    actor_id: str | Unset = UNSET
    actor_name: str | Unset = UNSET
    agent_id: str | Unset = UNSET
    request_id: str | Unset = UNSET
    resource_name: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        action = self.action.value

        changes = []
        for changes_item_data in self.changes:
            changes_item = changes_item_data.to_dict()
            changes.append(changes_item)

        created_at = self.created_at.isoformat()

        id = self.id

        resource_id = self.resource_id

        resource_type = self.resource_type.value

        source = self.source.value

        actor_id = self.actor_id

        actor_name = self.actor_name

        agent_id = self.agent_id

        request_id = self.request_id

        resource_name = self.resource_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "action": action,
                "changes": changes,
                "created_at": created_at,
                "id": id,
                "resource_id": resource_id,
                "resource_type": resource_type,
                "source": source,
            }
        )
        if actor_id is not UNSET:
            field_dict["actor_id"] = actor_id
        if actor_name is not UNSET:
            field_dict["actor_name"] = actor_name
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if request_id is not UNSET:
            field_dict["request_id"] = request_id
        if resource_name is not UNSET:
            field_dict["resource_name"] = resource_name

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.audit_change import AuditChange

        d = dict(src_dict)
        action = AuditAction(d.pop("action"))

        changes = []
        _changes = d.pop("changes")
        for changes_item_data in _changes:
            changes_item = AuditChange.from_dict(changes_item_data)

            changes.append(changes_item)

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        resource_id = d.pop("resource_id")

        resource_type = AuditResourceType(d.pop("resource_type"))

        source = AuditSource(d.pop("source"))

        actor_id = d.pop("actor_id", UNSET)

        actor_name = d.pop("actor_name", UNSET)

        agent_id = d.pop("agent_id", UNSET)

        request_id = d.pop("request_id", UNSET)

        resource_name = d.pop("resource_name", UNSET)

        audit_entry = cls(
            action=action,
            changes=changes,
            created_at=created_at,
            id=id,
            resource_id=resource_id,
            resource_type=resource_type,
            source=source,
            actor_id=actor_id,
            actor_name=actor_name,
            agent_id=agent_id,
            request_id=request_id,
            resource_name=resource_name,
        )

        audit_entry.additional_properties = d
        return audit_entry

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
