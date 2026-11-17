from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_tool import SessionTool


T = TypeVar("T", bound="ResumeSessionRequest")


@_attrs_define
class ResumeSessionRequest:
    """What a caller carrying a conversation on says about its tools, over what it ran with.

    Attributes:
        tool_timeout_ms (int | Unset): How long the model waits for the result of a tool the caller or a worker runs, in
            milliseconds, as on createSession, in place of what the conversation ran with. Omitted keeps that, which a
            conversation that ended keeps too. Zero is 30 seconds.
        tools (list[SessionTool] | Unset): The caller's own tools, in place of the ones the conversation was opened
            with, which it keeps when this is omitted. An empty list offers none. A conversation still running keeps its
            tools until it ends and is carried on.
    """

    tool_timeout_ms: int | Unset = UNSET
    tools: list[SessionTool] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        tool_timeout_ms = self.tool_timeout_ms

        tools: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = []
            for tools_item_data in self.tools:
                tools_item = tools_item_data.to_dict()
                tools.append(tools_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if tool_timeout_ms is not UNSET:
            field_dict["tool_timeout_ms"] = tool_timeout_ms
        if tools is not UNSET:
            field_dict["tools"] = tools

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.session_tool import SessionTool

        d = dict(src_dict)
        tool_timeout_ms = d.pop("tool_timeout_ms", UNSET)

        _tools = d.pop("tools", UNSET)
        tools: list[SessionTool] | Unset = UNSET
        if _tools is not UNSET:
            tools = []
            for tools_item_data in _tools:
                tools_item = SessionTool.from_dict(tools_item_data)

                tools.append(tools_item)

        resume_session_request = cls(
            tool_timeout_ms=tool_timeout_ms,
            tools=tools,
        )

        resume_session_request.additional_properties = d
        return resume_session_request

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
