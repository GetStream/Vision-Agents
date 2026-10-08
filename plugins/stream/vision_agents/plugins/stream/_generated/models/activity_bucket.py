from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ActivityBucket")


@_attrs_define
class ActivityBucket:
    """
    Attributes:
        active_users (int): Distinct end users who opened a session or asked something of an agent in the bucket. A
            guest who later turned out to be a known user counts as that user.
            A caller that named nobody is not counted, and neither is an anonymous one: an anonymous name is a claim nothing
            verified, so counting it would make guessing a name enough to inflate this.
        bucket (datetime.datetime):
        calls (int):
        messages (int): Responses the agents produced, which is one per thing asked of them.
        phone_minutes (float): The part of voice_minutes that arrived over a phone number.
        sessions (int):
        voice_minutes (float): How long those calls lasted. One still running counts up to now.
    """

    active_users: int
    bucket: datetime.datetime
    calls: int
    messages: int
    phone_minutes: float
    sessions: int
    voice_minutes: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        active_users = self.active_users

        bucket = self.bucket.isoformat()

        calls = self.calls

        messages = self.messages

        phone_minutes = self.phone_minutes

        sessions = self.sessions

        voice_minutes = self.voice_minutes

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "active_users": active_users,
                "bucket": bucket,
                "calls": calls,
                "messages": messages,
                "phone_minutes": phone_minutes,
                "sessions": sessions,
                "voice_minutes": voice_minutes,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        active_users = d.pop("active_users")

        bucket = datetime.datetime.fromisoformat(d.pop("bucket"))

        calls = d.pop("calls")

        messages = d.pop("messages")

        phone_minutes = d.pop("phone_minutes")

        sessions = d.pop("sessions")

        voice_minutes = d.pop("voice_minutes")

        activity_bucket = cls(
            active_users=active_users,
            bucket=bucket,
            calls=calls,
            messages=messages,
            phone_minutes=phone_minutes,
            sessions=sessions,
            voice_minutes=voice_minutes,
        )

        activity_bucket.additional_properties = d
        return activity_bucket

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
