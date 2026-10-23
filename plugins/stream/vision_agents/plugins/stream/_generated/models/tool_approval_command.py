from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.tool_approval_command_type import ToolApprovalCommandType
from ..types import UNSET, Unset

T = TypeVar("T", bound="ToolApprovalCommand")


@_attrs_define
class ToolApprovalCommand:
    """A person's answer to a call awaiting their approval, from a persistent text command. It changes only how the call is
    shown; the call still needs a tool_result. With allowed false and only tool_call_id it declines the sign-in a
    connector call waits for (awaiting_authorization) instead, and the router answers that call itself. For a connector
    call whose binding sets require_approval, the answer decides whether the router sends it, and names tool_call_id and
    request_id; turn_id may be left out.

        Attributes:
            allowed (bool):
            request_id (str):
            tool_call_id (str):
            type_ (ToolApprovalCommandType):
            summary (str | Unset): Shown on the declined call, such as "Location not shared".
            turn_id (str | Unset): The turn of the call. Required, except for a connector call whose binding sets
                require_approval.
    """

    allowed: bool
    request_id: str
    tool_call_id: str
    type_: ToolApprovalCommandType
    summary: str | Unset = UNSET
    turn_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allowed = self.allowed

        request_id = self.request_id

        tool_call_id = self.tool_call_id

        type_ = self.type_.value

        summary = self.summary

        turn_id = self.turn_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "allowed": allowed,
                "request_id": request_id,
                "tool_call_id": tool_call_id,
                "type": type_,
            }
        )
        if summary is not UNSET:
            field_dict["summary"] = summary
        if turn_id is not UNSET:
            field_dict["turn_id"] = turn_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        allowed = d.pop("allowed")

        request_id = d.pop("request_id")

        tool_call_id = d.pop("tool_call_id")

        type_ = ToolApprovalCommandType(d.pop("type"))

        summary = d.pop("summary", UNSET)

        turn_id = d.pop("turn_id", UNSET)

        tool_approval_command = cls(
            allowed=allowed,
            request_id=request_id,
            tool_call_id=tool_call_id,
            type_=type_,
            summary=summary,
            turn_id=turn_id,
        )

        tool_approval_command.additional_properties = d
        return tool_approval_command

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
