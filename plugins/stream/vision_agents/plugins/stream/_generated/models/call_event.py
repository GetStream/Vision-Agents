from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.decision_kind import DecisionKind
from ..types import UNSET, Unset

T = TypeVar("T", bound="CallEvent")


@_attrs_define
class CallEvent:
    """
    Attributes:
        at (datetime.datetime):
        kind (DecisionKind): What a conversation decided. Asking puts a settled turn to the flow controller; waiting
            leaves it because the caller has not finished; ignoring drops speech meant for somebody else; answering replies
            to it; queueing holds it until the agent has stopped talking; interrupting abandons the reply being spoken and
            shortening ends it early; a backchannel is a listening noise that never reaches the model; superseding drops a
            ruling about words that have since changed; compacting replaces old history with a summary; delegating hands
            work to the subagent and settling is that work coming back, answered or not.
        reason (str): Why the conversation chose it, in words.
        turn_id (str | Unset): The exchange it was about, which lines it up against that turn's timings.
        participant (str | Unset): Who it concerned.
        said (str | Unset): What was heard, what the agent decided to say, or what the subagent came back with.
        latency_ms (float | Unset): What the flow controller took to rule, or what the subagent took to answer. Zero
            where nothing was asked.
    """

    at: datetime.datetime
    kind: DecisionKind
    reason: str
    turn_id: str | Unset = UNSET
    participant: str | Unset = UNSET
    said: str | Unset = UNSET
    latency_ms: float | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        at = self.at.isoformat()

        kind = self.kind.value

        reason = self.reason

        turn_id = self.turn_id

        participant = self.participant

        said = self.said

        latency_ms = self.latency_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "at": at,
                "kind": kind,
                "reason": reason,
            }
        )
        if turn_id is not UNSET:
            field_dict["turn_id"] = turn_id
        if participant is not UNSET:
            field_dict["participant"] = participant
        if said is not UNSET:
            field_dict["said"] = said
        if latency_ms is not UNSET:
            field_dict["latency_ms"] = latency_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        at = datetime.datetime.fromisoformat(d.pop("at"))

        kind = DecisionKind(d.pop("kind"))

        reason = d.pop("reason")

        turn_id = d.pop("turn_id", UNSET)

        participant = d.pop("participant", UNSET)

        said = d.pop("said", UNSET)

        latency_ms = d.pop("latency_ms", UNSET)

        call_event = cls(
            at=at,
            kind=kind,
            reason=reason,
            turn_id=turn_id,
            participant=participant,
            said=said,
            latency_ms=latency_ms,
        )

        call_event.additional_properties = d
        return call_event

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
