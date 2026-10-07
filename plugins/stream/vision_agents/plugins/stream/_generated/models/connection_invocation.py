from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.invocation_error_type import InvocationErrorType
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectionInvocation")


@_attrs_define
class ConnectionInvocation:
    """One connector tool call a session ran through the connection: the binding, the tool, how long it took and how it
    failed. What the call was asked and answered is never kept.

        Attributes:
            binding (str): The alias the config binds the connector under.
            config_id (str): The agent config whose binding the call went through.
            connection_id (str):
            connector_id (str):
            id (str):
            latency_ms (int): From the call reaching the router to its answer, the router's own checks included.
            started_at (datetime.datetime):
            tool (str): The tool's name at the provider, without the alias.
            error_type (InvocationErrorType | Unset): customer_auth: the provider refused the connection's credential, or it
                had none; reconnect it. external_server: the provider answered with a failure or could not be reached.
                client_timeout: the router stopped waiting before the provider answered, and nothing says it got the call.
                outcome_unknown: the call was sent and cut off, by the binding's timeout or an interrupted turn, so it may have
                been done. denied: the router refused it before anything was sent.
            session_id (str | Unset): The session that called it. Absent for an incognito session, whose calls are tied to
                no conversation.
    """

    binding: str
    config_id: str
    connection_id: str
    connector_id: str
    id: str
    latency_ms: int
    started_at: datetime.datetime
    tool: str
    error_type: InvocationErrorType | Unset = UNSET
    session_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        binding = self.binding

        config_id = self.config_id

        connection_id = self.connection_id

        connector_id = self.connector_id

        id = self.id

        latency_ms = self.latency_ms

        started_at = self.started_at.isoformat()

        tool = self.tool

        error_type: str | Unset = UNSET
        if not isinstance(self.error_type, Unset):
            error_type = self.error_type.value

        session_id = self.session_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "binding": binding,
                "config_id": config_id,
                "connection_id": connection_id,
                "connector_id": connector_id,
                "id": id,
                "latency_ms": latency_ms,
                "started_at": started_at,
                "tool": tool,
            }
        )
        if error_type is not UNSET:
            field_dict["error_type"] = error_type
        if session_id is not UNSET:
            field_dict["session_id"] = session_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        binding = d.pop("binding")

        config_id = d.pop("config_id")

        connection_id = d.pop("connection_id")

        connector_id = d.pop("connector_id")

        id = d.pop("id")

        latency_ms = d.pop("latency_ms")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        tool = d.pop("tool")

        _error_type = d.pop("error_type", UNSET)
        error_type: InvocationErrorType | Unset
        if isinstance(_error_type, Unset):
            error_type = UNSET
        else:
            error_type = InvocationErrorType(_error_type)

        session_id = d.pop("session_id", UNSET)

        connection_invocation = cls(
            binding=binding,
            config_id=config_id,
            connection_id=connection_id,
            connector_id=connector_id,
            id=id,
            latency_ms=latency_ms,
            started_at=started_at,
            tool=tool,
            error_type=error_type,
            session_id=session_id,
        )

        connection_invocation.additional_properties = d
        return connection_invocation

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
