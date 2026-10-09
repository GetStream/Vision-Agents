from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.agent_connector_selection_type import AgentConnectorSelectionType
from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentConnectorSelection")


@_attrs_define
class AgentConnectorSelection:
    """Which connection a binding's tools are called through.

    Attributes:
        type_ (AgentConnectorSelectionType): fixed is the app's own connection named by connection_id, the same for
            every session. session is the connection the session's verified end user picks when the session is created,
            which has to be their own. When they pick none, it is their connection to the connector if exactly one of theirs
            is connected.
        connection_id (str | Unset): Required for fixed, and refused for session.
    """

    type_: AgentConnectorSelectionType
    connection_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_.value

        connection_id = self.connection_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
            }
        )
        if connection_id is not UNSET:
            field_dict["connection_id"] = connection_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        type_ = AgentConnectorSelectionType(d.pop("type"))

        connection_id = d.pop("connection_id", UNSET)

        agent_connector_selection = cls(
            type_=type_,
            connection_id=connection_id,
        )

        agent_connector_selection.additional_properties = d
        return agent_connector_selection

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
