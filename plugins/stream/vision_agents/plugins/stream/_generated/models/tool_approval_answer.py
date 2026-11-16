from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ToolApprovalAnswer")


@_attrs_define
class ToolApprovalAnswer:
    """A person's answer to a connector call waiting for their approval. An unknown field is refused rather than ignored.

    Attributes:
        allowed (bool): Whether the person allows the call.
        request_id (str): The command the reply answers: support_runtime.command_id on the reply's message.
        tool_call_id (str): The call's id, the id of its ai_tool_call step.
        summary (str | Unset): Shown on the declined call, such as "Not now".
    """

    allowed: bool
    request_id: str
    tool_call_id: str
    summary: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        allowed = self.allowed

        request_id = self.request_id

        tool_call_id = self.tool_call_id

        summary = self.summary

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "allowed": allowed,
                "request_id": request_id,
                "tool_call_id": tool_call_id,
            }
        )
        if summary is not UNSET:
            field_dict["summary"] = summary

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        allowed = d.pop("allowed")

        request_id = d.pop("request_id")

        tool_call_id = d.pop("tool_call_id")

        summary = d.pop("summary", UNSET)

        tool_approval_answer = cls(
            allowed=allowed,
            request_id=request_id,
            tool_call_id=tool_call_id,
            summary=summary,
        )

        return tool_approval_answer
