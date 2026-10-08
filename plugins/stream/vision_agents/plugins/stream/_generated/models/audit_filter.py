from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.audit_action import AuditAction
from ..models.audit_resource_type import AuditResourceType
from ..models.audit_source import AuditSource
from ..types import UNSET, Unset

T = TypeVar("T", bound="AuditFilter")


@_attrs_define
class AuditFilter:
    """Which changes to list. A field not listed here is refused rather than ignored.

    Attributes:
        action (AuditAction | Unset): created, updated or deleted for a change somebody made one at a time. synced is a
            whole agent directory written at once by POST /v1/agents/sync, and is what a later sync measures the edits made
            since against.
        agent_id (str | Unset): One agent's history: changes to the agent itself and to the skills and knowledge under
            it.
        resource_id (str | Unset): One resource's own history.
        resource_type (AuditResourceType | Unset): What the change was made to. All of them are configuration: what an
            agent does while it runs is traffic, and is read from the sessions and the logs instead.
        source (AuditSource | Unset): Which client made it, from the X-Stream-Client header. api is a caller that named
            no client: it reached the API directly, which is all that can be said about it.
    """

    action: AuditAction | Unset = UNSET
    agent_id: str | Unset = UNSET
    resource_id: str | Unset = UNSET
    resource_type: AuditResourceType | Unset = UNSET
    source: AuditSource | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        action: str | Unset = UNSET
        if not isinstance(self.action, Unset):
            action = self.action.value

        agent_id = self.agent_id

        resource_id = self.resource_id

        resource_type: str | Unset = UNSET
        if not isinstance(self.resource_type, Unset):
            resource_type = self.resource_type.value

        source: str | Unset = UNSET
        if not isinstance(self.source, Unset):
            source = self.source.value

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if action is not UNSET:
            field_dict["action"] = action
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if resource_id is not UNSET:
            field_dict["resource_id"] = resource_id
        if resource_type is not UNSET:
            field_dict["resource_type"] = resource_type
        if source is not UNSET:
            field_dict["source"] = source

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        _action = d.pop("action", UNSET)
        action: AuditAction | Unset
        if isinstance(_action, Unset):
            action = UNSET
        else:
            action = AuditAction(_action)

        agent_id = d.pop("agent_id", UNSET)

        resource_id = d.pop("resource_id", UNSET)

        _resource_type = d.pop("resource_type", UNSET)
        resource_type: AuditResourceType | Unset
        if isinstance(_resource_type, Unset):
            resource_type = UNSET
        else:
            resource_type = AuditResourceType(_resource_type)

        _source = d.pop("source", UNSET)
        source: AuditSource | Unset
        if isinstance(_source, Unset):
            source = UNSET
        else:
            source = AuditSource(_source)

        audit_filter = cls(
            action=action,
            agent_id=agent_id,
            resource_id=resource_id,
            resource_type=resource_type,
            source=source,
        )

        return audit_filter
