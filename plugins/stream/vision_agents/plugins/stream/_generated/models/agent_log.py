from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.agent_log_severity import AgentLogSeverity
from ..models.agent_log_source import AgentLogSource
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_log_details import AgentLogDetails


T = TypeVar("T", bound="AgentLog")


@_attrs_define
class AgentLog:
    """
    Attributes:
        agent_id (str):
        config_id (str):
        event_type (str):
        id (str):
        ingested_at (datetime.datetime):
        message (str):
        occurred_at (datetime.datetime):
        session_id (str):
        severity (AgentLogSeverity):
        source (AgentLogSource):
        cursor (str | Unset):
        details (AgentLogDetails | Unset):
        user_id (str | Unset):
    """

    agent_id: str
    config_id: str
    event_type: str
    id: str
    ingested_at: datetime.datetime
    message: str
    occurred_at: datetime.datetime
    session_id: str
    severity: AgentLogSeverity
    source: AgentLogSource
    cursor: str | Unset = UNSET
    details: AgentLogDetails | Unset = UNSET
    user_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent_id = self.agent_id

        config_id = self.config_id

        event_type = self.event_type

        id = self.id

        ingested_at = self.ingested_at.isoformat()

        message = self.message

        occurred_at = self.occurred_at.isoformat()

        session_id = self.session_id

        severity = self.severity.value

        source = self.source.value

        cursor = self.cursor

        details: dict[str, Any] | Unset = UNSET
        if not isinstance(self.details, Unset):
            details = self.details.to_dict()

        user_id = self.user_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "agent_id": agent_id,
                "config_id": config_id,
                "event_type": event_type,
                "id": id,
                "ingested_at": ingested_at,
                "message": message,
                "occurred_at": occurred_at,
                "session_id": session_id,
                "severity": severity,
                "source": source,
            }
        )
        if cursor is not UNSET:
            field_dict["cursor"] = cursor
        if details is not UNSET:
            field_dict["details"] = details
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_log_details import AgentLogDetails

        d = dict(src_dict)
        agent_id = d.pop("agent_id")

        config_id = d.pop("config_id")

        event_type = d.pop("event_type")

        id = d.pop("id")

        ingested_at = datetime.datetime.fromisoformat(d.pop("ingested_at"))

        message = d.pop("message")

        occurred_at = datetime.datetime.fromisoformat(d.pop("occurred_at"))

        session_id = d.pop("session_id")

        severity = AgentLogSeverity(d.pop("severity"))

        source = AgentLogSource(d.pop("source"))

        cursor = d.pop("cursor", UNSET)

        _details = d.pop("details", UNSET)
        details: AgentLogDetails | Unset
        if isinstance(_details, Unset):
            details = UNSET
        else:
            details = AgentLogDetails.from_dict(_details)

        user_id = d.pop("user_id", UNSET)

        agent_log = cls(
            agent_id=agent_id,
            config_id=config_id,
            event_type=event_type,
            id=id,
            ingested_at=ingested_at,
            message=message,
            occurred_at=occurred_at,
            session_id=session_id,
            severity=severity,
            source=source,
            cursor=cursor,
            details=details,
            user_id=user_id,
        )

        agent_log.additional_properties = d
        return agent_log

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
