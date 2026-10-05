from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="PhoneSandbox")


@_attrs_define
class PhoneSandbox:
    """What an app may text and call before a 10DLC use case of its is approved.

    Attributes:
        audio_minutes_per_day (int):
        audio_seconds_today (int): Seconds of outbound calls since midnight UTC.
        enabled (bool): Whether this deployment sandboxes apps with no approved use case. Off on a self-hosted router.
        max_recipients (int):
        messages_per_day (int):
        messages_today (int): Messages sent since midnight UTC.
        recipients (list[str]): The only numbers a sandboxed app may text and call.
        sandboxed (bool): Whether this app is held to the limits below: true until one of its use cases is approved.
    """

    audio_minutes_per_day: int
    audio_seconds_today: int
    enabled: bool
    max_recipients: int
    messages_per_day: int
    messages_today: int
    recipients: list[str]
    sandboxed: bool
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        audio_minutes_per_day = self.audio_minutes_per_day

        audio_seconds_today = self.audio_seconds_today

        enabled = self.enabled

        max_recipients = self.max_recipients

        messages_per_day = self.messages_per_day

        messages_today = self.messages_today

        recipients = self.recipients

        sandboxed = self.sandboxed

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "audio_minutes_per_day": audio_minutes_per_day,
                "audio_seconds_today": audio_seconds_today,
                "enabled": enabled,
                "max_recipients": max_recipients,
                "messages_per_day": messages_per_day,
                "messages_today": messages_today,
                "recipients": recipients,
                "sandboxed": sandboxed,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        audio_minutes_per_day = d.pop("audio_minutes_per_day")

        audio_seconds_today = d.pop("audio_seconds_today")

        enabled = d.pop("enabled")

        max_recipients = d.pop("max_recipients")

        messages_per_day = d.pop("messages_per_day")

        messages_today = d.pop("messages_today")

        recipients = cast(list[str], d.pop("recipients"))

        sandboxed = d.pop("sandboxed")

        phone_sandbox = cls(
            audio_minutes_per_day=audio_minutes_per_day,
            audio_seconds_today=audio_seconds_today,
            enabled=enabled,
            max_recipients=max_recipients,
            messages_per_day=messages_per_day,
            messages_today=messages_today,
            recipients=recipients,
            sandboxed=sandboxed,
        )

        phone_sandbox.additional_properties = d
        return phone_sandbox

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
