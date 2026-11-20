from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.tool_call_frame_type import ToolCallFrameType
from ..models.tool_call_status import ToolCallStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.tool_call_state_custom import ToolCallStateCustom


T = TypeVar("T", bound="ToolCallState")


@_attrs_define
class ToolCallState:
    """One change to a tool call's state. Every watcher is sent tool_call.started once, tool_call.progress for each change
    while it runs or waits, and tool_call.finished once, carrying the call's whole state each time, so a client keeps
    the last one it got per tool_call_id.

        Attributes:
            duration_ms (int): From started_at to this change; on tool_call.finished, how long the call took.
            started_at (datetime.datetime):
            status (ToolCallStatus): in_progress while it runs, waiting while it is held for a person's approval, and
                success, failed or cancelled once it finished. A call declined, or whose approval nobody gave in time, is
                cancelled with rejected_at.
            tool (str):
            tool_call_id (str): The call, as tool_call, tool_started and tool_ran name it.
            turn_id (str):
            type_ (ToolCallFrameType): tool_call.started is the call's first frame, tool_call.progress one while it runs or
                waits, and tool_call.finished its last.
            approved_at (datetime.datetime | Unset): When a person allowed the call.
            error (str | Unset): Why a failed call failed, or what a declined one shows, such as "Not allowed in time".
            progress (str | Unset): The last thing said about the call while it runs, such as "Searching for flights": a
                tool_update, the connector binding's pre_speech, or the question it waits on.
            rejected_at (datetime.datetime | Unset): When a person declined the call, or its wait for them passed.
            display_title (str | Unset): What the call is doing, in words for the people in the conversation, such as
                "Checking the weather": the display_title its tool was declared with, or "Searching the web" for the router's
                own search. Absent when the tool declared none.
            custom (ToolCallStateCustom | Unset): The data whoever runs the call last sent with a tool_update, such as what
                a card under the message shows. Each update with custom replaces it whole, and it is kept once the call
                finishes.
    """

    duration_ms: int
    started_at: datetime.datetime
    status: ToolCallStatus
    tool: str
    tool_call_id: str
    turn_id: str
    type_: ToolCallFrameType
    approved_at: datetime.datetime | Unset = UNSET
    error: str | Unset = UNSET
    progress: str | Unset = UNSET
    rejected_at: datetime.datetime | Unset = UNSET
    display_title: str | Unset = UNSET
    custom: ToolCallStateCustom | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        duration_ms = self.duration_ms

        started_at = self.started_at.isoformat()

        status = self.status.value

        tool = self.tool

        tool_call_id = self.tool_call_id

        turn_id = self.turn_id

        type_ = self.type_.value

        approved_at: str | Unset = UNSET
        if not isinstance(self.approved_at, Unset):
            approved_at = self.approved_at.isoformat()

        error = self.error

        progress = self.progress

        rejected_at: str | Unset = UNSET
        if not isinstance(self.rejected_at, Unset):
            rejected_at = self.rejected_at.isoformat()

        display_title = self.display_title

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "duration_ms": duration_ms,
                "started_at": started_at,
                "status": status,
                "tool": tool,
                "tool_call_id": tool_call_id,
                "turn_id": turn_id,
                "type": type_,
            }
        )
        if approved_at is not UNSET:
            field_dict["approved_at"] = approved_at
        if error is not UNSET:
            field_dict["error"] = error
        if progress is not UNSET:
            field_dict["progress"] = progress
        if rejected_at is not UNSET:
            field_dict["rejected_at"] = rejected_at
        if display_title is not UNSET:
            field_dict["display_title"] = display_title
        if custom is not UNSET:
            field_dict["custom"] = custom

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.tool_call_state_custom import ToolCallStateCustom

        d = dict(src_dict)
        duration_ms = d.pop("duration_ms")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        status = ToolCallStatus._decode(d.pop("status"))

        tool = d.pop("tool")

        tool_call_id = d.pop("tool_call_id")

        turn_id = d.pop("turn_id")

        type_ = ToolCallFrameType._decode(d.pop("type"))

        _approved_at = d.pop("approved_at", UNSET)
        approved_at: datetime.datetime | Unset
        if isinstance(_approved_at, Unset):
            approved_at = UNSET
        else:
            approved_at = datetime.datetime.fromisoformat(_approved_at)

        error = d.pop("error", UNSET)

        progress = d.pop("progress", UNSET)

        _rejected_at = d.pop("rejected_at", UNSET)
        rejected_at: datetime.datetime | Unset
        if isinstance(_rejected_at, Unset):
            rejected_at = UNSET
        else:
            rejected_at = datetime.datetime.fromisoformat(_rejected_at)

        display_title = d.pop("display_title", UNSET)

        _custom = d.pop("custom", UNSET)
        custom: ToolCallStateCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = ToolCallStateCustom.from_dict(_custom)

        tool_call_state = cls(
            duration_ms=duration_ms,
            started_at=started_at,
            status=status,
            tool=tool,
            tool_call_id=tool_call_id,
            turn_id=turn_id,
            type_=type_,
            approved_at=approved_at,
            error=error,
            progress=progress,
            rejected_at=rejected_at,
            display_title=display_title,
            custom=custom,
        )

        tool_call_state.additional_properties = d
        return tool_call_state

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
