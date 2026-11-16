from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="LeftOutConnector")


@_attrs_define
class LeftOutConnector:
    """A connector agent.yaml names that a sync left out, because the app has no connection by the id or label it names.

    Attributes:
        connection (str): The connection id or label agent.yaml named, which no connection of the app has.
        connector_id (str): The connector agent.yaml named, or the alias when it named none.
        name (str): The alias agent.yaml gave the connector.
    """

    connection: str
    connector_id: str
    name: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connection = self.connection

        connector_id = self.connector_id

        name = self.name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection": connection,
                "connector_id": connector_id,
                "name": name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        connection = d.pop("connection")

        connector_id = d.pop("connector_id")

        name = d.pop("name")

        left_out_connector = cls(
            connection=connection,
            connector_id=connector_id,
            name=name,
        )

        left_out_connector.additional_properties = d
        return left_out_connector

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
