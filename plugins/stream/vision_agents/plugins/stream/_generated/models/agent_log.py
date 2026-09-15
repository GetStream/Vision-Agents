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
        id (str):
        config_id (str):
        agent_id (str):
        session_id (str):
        source (AgentLogSource):
        severity (AgentLogSeverity):
        event_type (str):
        message (str):
        occurred_at (datetime.datetime):
        ingested_at (datetime.datetime):
        cursor (str | Unset):
        user_id (str | Unset):
        details (AgentLogDetails | Unset):
    """

    id: str
    config_id: str
    agent_id: str
    session_id: str
    source: AgentLogSource
    severity: AgentLogSeverity
    event_type: str
    message: str
    occurred_at: datetime.datetime
    ingested_at: datetime.datetime
    cursor: str | Unset = UNSET
    user_id: str | Unset = UNSET
    details: AgentLogDetails | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        config_id = self.config_id

        agent_id = self.agent_id

        session_id = self.session_id

        source = self.source.value

        severity = self.severity.value

        event_type = self.event_type

        message = self.message

        occurred_at = self.occurred_at.isoformat()

        ingested_at = self.ingested_at.isoformat()

        cursor = self.cursor

        user_id = self.user_id

        details: dict[str, Any] | Unset = UNSET
        if not isinstance(self.details, Unset):
            details = self.details.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "config_id": config_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "source": source,
                "severity": severity,
                "event_type": event_type,
                "message": message,
                "occurred_at": occurred_at,
                "ingested_at": ingested_at,
            }
        )
        if cursor is not UNSET:
            field_dict["cursor"] = cursor
        if user_id is not UNSET:
            field_dict["user_id"] = user_id
        if details is not UNSET:
            field_dict["details"] = details

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_log_details import AgentLogDetails

        d = dict(src_dict)
        id = d.pop("id")

        config_id = d.pop("config_id")

        agent_id = d.pop("agent_id")

        session_id = d.pop("session_id")

        source = AgentLogSource(d.pop("source"))

        severity = AgentLogSeverity(d.pop("severity"))

        event_type = d.pop("event_type")

        message = d.pop("message")

        occurred_at = datetime.datetime.fromisoformat(d.pop("occurred_at"))

        ingested_at = datetime.datetime.fromisoformat(d.pop("ingested_at"))

        cursor = d.pop("cursor", UNSET)

        user_id = d.pop("user_id", UNSET)

        _details = d.pop("details", UNSET)
        details: AgentLogDetails | Unset
        if isinstance(_details, Unset):
            details = UNSET
        else:
            details = AgentLogDetails.from_dict(_details)

        agent_log = cls(
            id=id,
            config_id=config_id,
            agent_id=agent_id,
            session_id=session_id,
            source=source,
            severity=severity,
            event_type=event_type,
            message=message,
            occurred_at=occurred_at,
            ingested_at=ingested_at,
            cursor=cursor,
            user_id=user_id,
            details=details,
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
