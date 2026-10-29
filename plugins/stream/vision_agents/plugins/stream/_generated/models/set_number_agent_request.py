from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SetNumberAgentRequest")


@_attrs_define
class SetNumberAgentRequest:
    """
    Attributes:
        agent (str | Unset): The name of the agent config that answers the number. Give this or config_id.
        config_id (str | Unset): The id of the agent config that answers the number. Give this or agent.
    """

    agent: str | Unset = UNSET
    config_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent = self.agent

        config_id = self.config_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if agent is not UNSET:
            field_dict["agent"] = agent
        if config_id is not UNSET:
            field_dict["config_id"] = config_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        agent = d.pop("agent", UNSET)

        config_id = d.pop("config_id", UNSET)

        set_number_agent_request = cls(
            agent=agent,
            config_id=config_id,
        )

        set_number_agent_request.additional_properties = d
        return set_number_agent_request

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
