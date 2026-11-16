from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentTools")


@_attrs_define
class AgentTools:
    """How an agent is offered its tools, and how long what they were given and gave back is kept.

    Attributes:
        output_retention (str | Unset): How long this agent's sessions keep what any of its tools were given and gave
            back on the turns they record: a tool's arguments, its result, a search's query and the wording of a tool's
            error. none records none of it; a duration such as 12h or 30d (at least 1h, at most 3650d) keeps it that long
            after the call and then takes it out, within minutes. The item stays, with its tool name, its time, its call id
            and whether it failed. Left out on a create, or empty, it is kept as long as the session; left out on an update,
            the stored setting stays, and an empty string removes it. The app's and the organization's policies
            (tool_output_retention) can shorten it and never lengthen it, and the shortest applies. It applies to tool calls
            recorded after it is set. Example: 30d.
        progressive (bool | Unset): Offer each tool by the first line of its description, with its arguments'
            descriptions left out, and have the first call to a tool return its full description and input schema instead of
            running it. It saves context on an agent with many tools, at the cost of one more model turn for each tool a
            conversation uses. Off by default. Left out on an update, the stored setting stays.
    """

    output_retention: str | Unset = UNSET
    progressive: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        output_retention = self.output_retention

        progressive = self.progressive

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if output_retention is not UNSET:
            field_dict["output_retention"] = output_retention
        if progressive is not UNSET:
            field_dict["progressive"] = progressive

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        output_retention = d.pop("output_retention", UNSET)

        progressive = d.pop("progressive", UNSET)

        agent_tools = cls(
            output_retention=output_retention,
            progressive=progressive,
        )

        agent_tools.additional_properties = d
        return agent_tools

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
