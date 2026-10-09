from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.invocation_error_type import InvocationErrorType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.invocation_argument import InvocationArgument


T = TypeVar("T", bound="ConnectionInvocation")


@_attrs_define
class ConnectionInvocation:
    """One connector tool call a session ran through the connection: the binding, the tool, the shape of its arguments, how
    long it took and how it failed. No value the call was asked, and nothing it answered, is kept.

        Attributes:
            binding (str): The alias the config binds the connector under.
            config_id (str): The agent config whose binding the call went through.
            connection_id (str):
            connector_id (str):
            id (str):
            latency_ms (int): From the call reaching the router to its answer, the router's own checks included.
            started_at (datetime.datetime):
            tool (str): The tool's name at the provider, without the alias.
            arguments (list[InvocationArgument] | None | Unset): The shape of what the call was asked, sorted by name.
                Absent for a call asked with no arguments, for an incognito session's call, for arguments that were not a JSON
                object, and for a call recorded before the router kept it.
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
    arguments: list[InvocationArgument] | None | Unset = UNSET
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

        arguments: list[dict[str, Any]] | None | Unset
        if isinstance(self.arguments, Unset):
            arguments = UNSET
        elif isinstance(self.arguments, list):
            arguments = []
            for arguments_type_0_item_data in self.arguments:
                arguments_type_0_item = arguments_type_0_item_data.to_dict()
                arguments.append(arguments_type_0_item)

        else:
            arguments = self.arguments

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
        if arguments is not UNSET:
            field_dict["arguments"] = arguments
        if error_type is not UNSET:
            field_dict["error_type"] = error_type
        if session_id is not UNSET:
            field_dict["session_id"] = session_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.invocation_argument import InvocationArgument

        d = dict(src_dict)
        binding = d.pop("binding")

        config_id = d.pop("config_id")

        connection_id = d.pop("connection_id")

        connector_id = d.pop("connector_id")

        id = d.pop("id")

        latency_ms = d.pop("latency_ms")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        tool = d.pop("tool")

        def _parse_arguments(data: object) -> list[InvocationArgument] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                arguments_type_0 = []
                _arguments_type_0 = data
                for arguments_type_0_item_data in _arguments_type_0:
                    arguments_type_0_item = InvocationArgument.from_dict(
                        arguments_type_0_item_data
                    )

                    arguments_type_0.append(arguments_type_0_item)

                return arguments_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[InvocationArgument] | None | Unset, data)

        arguments = _parse_arguments(d.pop("arguments", UNSET))

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
            arguments=arguments,
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
