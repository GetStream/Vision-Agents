from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.tool_loading import ToolLoading
from ..types import UNSET, Unset

T = TypeVar("T", bound="ToolGroup")


@_attrs_define
class ToolGroup:
    """A named set of functions an SDK provides that the agent may offer, how they are loaded and how long what they were
    given and gave back is kept. A function an SDK registers, declares on a session or hosts in a worker is offered only
    when a group names it.

        Attributes:
            name (str): What the group is called, unique within the config.
            tools (list[str]): The functions' names, as the SDK registers them. A name sits in one group at most.
            loading (ToolLoading | Unset): How a tool is offered to the model. full sends its whole description and input
                schema from the start. summary sends the first line of its description and its arguments without their
                descriptions, and the first call returns the full description and schema instead of running it: one more model
                turn for each tool a conversation uses, for less context on every turn. deferred offers nothing until the model
                needs it: an agent with a deferred tool is offered search_tool, which finds deferred tools by what they do, and
                load_tool, which returns their full descriptions and offers them from the model's next step on.
            output_retention (str | Unset): How long the agent's sessions keep what the group's tools were given and gave
                back on the turns they record: a tool's arguments, its result and the wording of its error. none records none of
                it; a duration such as 12h or 30d (at least 1h, at most 3650d) keeps it that long after the call and then takes
                it out, within minutes. The item stays, with its tool name, its time, its call id and whether it failed. Left
                out or empty, it is kept as long as the session. The app's and the organization's policies
                (tool_output_retention) can shorten it and never lengthen it. It applies to tool calls recorded after it is set.
                Example: 30d.
    """

    name: str
    tools: list[str]
    loading: ToolLoading | Unset = UNSET
    output_retention: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        tools = self.tools

        loading: str | Unset = UNSET
        if not isinstance(self.loading, Unset):
            loading = self.loading.value

        output_retention = self.output_retention

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "tools": tools,
            }
        )
        if loading is not UNSET:
            field_dict["loading"] = loading
        if output_retention is not UNSET:
            field_dict["output_retention"] = output_retention

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        tools = cast(list[str], d.pop("tools"))

        _loading = d.pop("loading", UNSET)
        loading: ToolLoading | Unset
        if isinstance(_loading, Unset):
            loading = UNSET
        else:
            loading = ToolLoading._decode(_loading)

        output_retention = d.pop("output_retention", UNSET)

        tool_group = cls(
            name=name,
            tools=tools,
            loading=loading,
            output_retention=output_retention,
        )

        tool_group.additional_properties = d
        return tool_group

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
