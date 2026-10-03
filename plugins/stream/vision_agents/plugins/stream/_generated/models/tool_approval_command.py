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
    shown; the call still needs a tool_result.

        Attributes:
            allowed (bool):
            command_id (str):
            tool_call_id (str):
            turn_id (str):
            type_ (ToolApprovalCommandType):
            summary (str | Unset): Shown on the declined call, such as "Location not shared".
    """

    allowed: bool
    command_id: str
    tool_call_id: str
    turn_id: str
    type_: ToolApprovalCommandType
    summary: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allowed = self.allowed

        command_id = self.command_id

        tool_call_id = self.tool_call_id

        turn_id = self.turn_id

        type_ = self.type_.value

        summary = self.summary

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "allowed": allowed,
                "command_id": command_id,
                "tool_call_id": tool_call_id,
                "turn_id": turn_id,
                "type": type_,
            }
        )
        if summary is not UNSET:
            field_dict["summary"] = summary

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        allowed = d.pop("allowed")

        command_id = d.pop("command_id")

        tool_call_id = d.pop("tool_call_id")

        turn_id = d.pop("turn_id")

        type_ = ToolApprovalCommandType(d.pop("type"))

        summary = d.pop("summary", UNSET)

        tool_approval_command = cls(
            allowed=allowed,
            command_id=command_id,
            tool_call_id=tool_call_id,
            turn_id=turn_id,
            type_=type_,
            summary=summary,
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
