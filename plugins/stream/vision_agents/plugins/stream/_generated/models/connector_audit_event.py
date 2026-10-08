from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connection_owner_type import ConnectionOwnerType
from ..models.connector_audit_action import ConnectorAuditAction
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorAuditEvent")


@_attrs_define
class ConnectorAuditEvent:
    """One grant a connection got, renewed or lost, one export of its access credential, or one direct call sent through
    it, with the ids that tie it to what caused it. It names no user and no provider account, so it outlives a user's
    connections being deleted.

        Attributes:
            action (ConnectorAuditAction): grant_created: a consent or a credentials write gave the connection a grant.
                grant_refreshed: the router renewed its credential. grant_revoked: the grant ended, because the provider refused
                or revoked it or the connection was deleted. token_export: the app's backend exported its access credential.
                proxy_call: a direct call went to the provider through the connection.
            connection_id (str): The connection, which may since have been deleted.
            connector_id (str):
            created_at (datetime.datetime):
            id (str):
            owner_type (ConnectionOwnerType): app is the app's own account, user one user's.
            attempt_id (str | Unset): The authorization attempt a consent finished. Absent once the connection's user was
                deleted.
            latency_ms (int | Unset): How long a proxy_call took until the provider's answer, in milliseconds. Absent for a
                grant.
            reason (str | Unset): Why: consent or credentials for a created grant; deleted or user_deleted for a delete; for
                a grant the provider ended, its word for why, such as invalid_grant, scope_required or revoked.
            request_id (str | Unset): The X-Request-Id of the API request that caused it. For a change a session's tool call
                caused, that is the request that created the session, not the one that asked for the turn. Absent for an
                incognito session's, and once the connection's user was deleted.
            revision (int | Unset): The connection's credential revision once the change was made. Absent when the change
                names none, as a delete.
            session_id (str | Unset): The session whose tool call caused it. Absent for an incognito session, and once the
                connection's user was deleted.
            status_code (int | Unset): A proxy_call's status from the provider. Absent when no answer came, and for a grant.
            target (str | Unset): The host a proxy_call reached. Absent for a grant.
    """

    action: ConnectorAuditAction
    connection_id: str
    connector_id: str
    created_at: datetime.datetime
    id: str
    owner_type: ConnectionOwnerType
    attempt_id: str | Unset = UNSET
    latency_ms: int | Unset = UNSET
    reason: str | Unset = UNSET
    request_id: str | Unset = UNSET
    revision: int | Unset = UNSET
    session_id: str | Unset = UNSET
    status_code: int | Unset = UNSET
    target: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        action = self.action.value

        connection_id = self.connection_id

        connector_id = self.connector_id

        created_at = self.created_at.isoformat()

        id = self.id

        owner_type = self.owner_type.value

        attempt_id = self.attempt_id

        latency_ms = self.latency_ms

        reason = self.reason

        request_id = self.request_id

        revision = self.revision

        session_id = self.session_id

        status_code = self.status_code

        target = self.target

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "action": action,
                "connection_id": connection_id,
                "connector_id": connector_id,
                "created_at": created_at,
                "id": id,
                "owner_type": owner_type,
            }
        )
        if attempt_id is not UNSET:
            field_dict["attempt_id"] = attempt_id
        if latency_ms is not UNSET:
            field_dict["latency_ms"] = latency_ms
        if reason is not UNSET:
            field_dict["reason"] = reason
        if request_id is not UNSET:
            field_dict["request_id"] = request_id
        if revision is not UNSET:
            field_dict["revision"] = revision
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
        if status_code is not UNSET:
            field_dict["status_code"] = status_code
        if target is not UNSET:
            field_dict["target"] = target

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        action = ConnectorAuditAction(d.pop("action"))

        connection_id = d.pop("connection_id")

        connector_id = d.pop("connector_id")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        owner_type = ConnectionOwnerType(d.pop("owner_type"))

        attempt_id = d.pop("attempt_id", UNSET)

        latency_ms = d.pop("latency_ms", UNSET)

        reason = d.pop("reason", UNSET)

        request_id = d.pop("request_id", UNSET)

        revision = d.pop("revision", UNSET)

        session_id = d.pop("session_id", UNSET)

        status_code = d.pop("status_code", UNSET)

        target = d.pop("target", UNSET)

        connector_audit_event = cls(
            action=action,
            connection_id=connection_id,
            connector_id=connector_id,
            created_at=created_at,
            id=id,
            owner_type=owner_type,
            attempt_id=attempt_id,
            latency_ms=latency_ms,
            reason=reason,
            request_id=request_id,
            revision=revision,
            session_id=session_id,
            status_code=status_code,
            target=target,
        )

        connector_audit_event.additional_properties = d
        return connector_audit_event

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
