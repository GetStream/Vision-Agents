from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

T = TypeVar("T", bound="SessionConnectorBinding")


@_attrs_define
class SessionConnectorBinding:
    """The connection a session uses for one of its agent config's connector bindings chosen per session. Only a reference:
    the credential stays sealed on the connection.

        Attributes:
            connection_id (str): The caller's own connection to the binding's connector.
            name (str): The binding's alias in the agent config.
    """

    connection_id: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        connection_id = self.connection_id

        name = self.name

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "connection_id": connection_id,
                "name": name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        name = d.pop("name")

        session_connector_binding = cls(
            connection_id=connection_id,
            name=name,
        )

        return session_connector_binding
