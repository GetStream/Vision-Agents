from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ContactAttempt")


@_attrs_define
class ContactAttempt:
    """
    Attributes:
        attempt (int): Which attempt of the sequence, from 1.
        dial (int): 1, or 2 for the second call of a double dial.
        result (str): answered, unanswered, voicemail, invalid_number, refused, deferred, paused, canceled or abandoned;
            empty while the call is happening. Only answered, unanswered and voicemail placed a call that ended.
        started_at (datetime.datetime):
        call_id (str | Unset):
        ended_at (datetime.datetime | Unset):
        outcome (str | Unset): The outcome an answered call was read as.
    """

    attempt: int
    dial: int
    result: str
    started_at: datetime.datetime
    call_id: str | Unset = UNSET
    ended_at: datetime.datetime | Unset = UNSET
    outcome: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        attempt = self.attempt

        dial = self.dial

        result = self.result

        started_at = self.started_at.isoformat()

        call_id = self.call_id

        ended_at: str | Unset = UNSET
        if not isinstance(self.ended_at, Unset):
            ended_at = self.ended_at.isoformat()

        outcome = self.outcome

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "attempt": attempt,
                "dial": dial,
                "result": result,
                "started_at": started_at,
            }
        )
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if ended_at is not UNSET:
            field_dict["ended_at"] = ended_at
        if outcome is not UNSET:
            field_dict["outcome"] = outcome

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        attempt = d.pop("attempt")

        dial = d.pop("dial")

        result = d.pop("result")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        call_id = d.pop("call_id", UNSET)

        _ended_at = d.pop("ended_at", UNSET)
        ended_at: datetime.datetime | Unset
        if isinstance(_ended_at, Unset):
            ended_at = UNSET
        else:
            ended_at = datetime.datetime.fromisoformat(_ended_at)

        outcome = d.pop("outcome", UNSET)

        contact_attempt = cls(
            attempt=attempt,
            dial=dial,
            result=result,
            started_at=started_at,
            call_id=call_id,
            ended_at=ended_at,
            outcome=outcome,
        )

        contact_attempt.additional_properties = d
        return contact_attempt

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
