from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.call_outcome import CallOutcome
from ..types import UNSET, Unset

T = TypeVar("T", bound="SessionCallOutcome")


@_attrs_define
class SessionCallOutcome:
    """How the call the router placed for a session ended.

    Attributes:
        answered_at (datetime.datetime | Unset): When the person picked up: their phone joined the call.
        duration_seconds (int | Unset): How long they talked, from answered_at to when their phone left, rounded to the
            nearest second. Absent until it has.
        outcome (CallOutcome | Unset): How a placed call ended.
            completed: the person answered and the call ended.
            busy: the line was busy or the person declined.
            no_answer: it rang and nobody answered.
            failed: the vendor could not place it, for a bad number or a carrier fault.
            canceled: the vendor says the call was hung up before an answer.
            unknown: the vendor reports nothing about how it ended, and the phone never joined.
    """

    answered_at: datetime.datetime | Unset = UNSET
    duration_seconds: int | Unset = UNSET
    outcome: CallOutcome | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        answered_at: str | Unset = UNSET
        if not isinstance(self.answered_at, Unset):
            answered_at = self.answered_at.isoformat()

        duration_seconds = self.duration_seconds

        outcome: str | Unset = UNSET
        if not isinstance(self.outcome, Unset):
            outcome = self.outcome.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if answered_at is not UNSET:
            field_dict["answered_at"] = answered_at
        if duration_seconds is not UNSET:
            field_dict["duration_seconds"] = duration_seconds
        if outcome is not UNSET:
            field_dict["outcome"] = outcome

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        _answered_at = d.pop("answered_at", UNSET)
        answered_at: datetime.datetime | Unset
        if isinstance(_answered_at, Unset):
            answered_at = UNSET
        else:
            answered_at = datetime.datetime.fromisoformat(_answered_at)

        duration_seconds = d.pop("duration_seconds", UNSET)

        _outcome = d.pop("outcome", UNSET)
        outcome: CallOutcome | Unset
        if isinstance(_outcome, Unset):
            outcome = UNSET
        else:
            outcome = CallOutcome(_outcome)

        session_call_outcome = cls(
            answered_at=answered_at,
            duration_seconds=duration_seconds,
            outcome=outcome,
        )

        session_call_outcome.additional_properties = d
        return session_call_outcome

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
