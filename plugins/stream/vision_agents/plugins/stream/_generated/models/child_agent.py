from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ChildAgent")


@_attrs_define
class ChildAgent:
    """Another of the app's text agents a text agent may hand work to, offered to its model as a tool.

    Attributes:
        config (str): The agent config to hand work to: one of the app's text agents, by name or id. It is stored, and
            read back, by name.
        tool (str): The name the model calls it by, unique among the config's agents.
        description (str | Unset): What the model is told the agent does, which is the whole of how it decides when to
            call it.
        timeout_ms (int | Unset): How long one call may take, two minutes when left out. A call never outlasts the run
            that made it: it gets this or what is left of the run, whichever is less. A call that runs out of time, or whose
            agent fails, is answered to the model as a failed tool call, and the run goes on.
        max_tool_rounds (int | Unset): The reply to a tool result, counted since the agent was handed its input, that is
            offered no more tools and made to answer from what it has. Zero is the default of 8; a negative number never
            withdraws them, for an agent that chains many lookups.
    """

    config: str
    tool: str
    description: str | Unset = UNSET
    timeout_ms: int | Unset = UNSET
    max_tool_rounds: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        config = self.config

        tool = self.tool

        description = self.description

        timeout_ms = self.timeout_ms

        max_tool_rounds = self.max_tool_rounds

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "config": config,
                "tool": tool,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if timeout_ms is not UNSET:
            field_dict["timeout_ms"] = timeout_ms
        if max_tool_rounds is not UNSET:
            field_dict["max_tool_rounds"] = max_tool_rounds

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        config = d.pop("config")

        tool = d.pop("tool")

        description = d.pop("description", UNSET)

        timeout_ms = d.pop("timeout_ms", UNSET)

        max_tool_rounds = d.pop("max_tool_rounds", UNSET)

        child_agent = cls(
            config=config,
            tool=tool,
            description=description,
            timeout_ms=timeout_ms,
            max_tool_rounds=max_tool_rounds,
        )

        child_agent.additional_properties = d
        return child_agent

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
