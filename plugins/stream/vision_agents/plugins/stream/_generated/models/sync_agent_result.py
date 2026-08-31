from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.agent_config import AgentConfig


T = TypeVar("T", bound="SyncAgentResult")


@_attrs_define
class SyncAgentResult:
    """
    Attributes:
        unchanged (bool): True when the hash matched and nothing was written.
        config (AgentConfig):
    """

    unchanged: bool
    config: AgentConfig
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        unchanged = self.unchanged

        config = self.config.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "unchanged": unchanged,
                "config": config,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_config import AgentConfig

        d = dict(src_dict)
        unchanged = d.pop("unchanged")

        config = AgentConfig.from_dict(d.pop("config"))

        sync_agent_result = cls(
            unchanged=unchanged,
            config=config,
        )

        sync_agent_result.additional_properties = d
        return sync_agent_result

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
