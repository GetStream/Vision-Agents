from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.voicemail_action import VoicemailAction
from ..types import UNSET, Unset

T = TypeVar("T", bound="SessionVoicemail")


@_attrs_define
class SessionVoicemail:
    """What the agent does when an answering machine picks up.

    Attributes:
        action (VoicemailAction): hang_up: end the call at once, saying nothing.
            leave_message: say message once the greeting has ended, then hang up.
        message (str | Unset): Said word for word once the machine's greeting has ended, then the agent hangs up.
            Required for leave_message, and not only spaces. Ignored for hang_up.
    """

    action: VoicemailAction
    message: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        action = self.action.value

        message = self.message

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "action": action,
            }
        )
        if message is not UNSET:
            field_dict["message"] = message

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        action = VoicemailAction(d.pop("action"))

        message = d.pop("message", UNSET)

        session_voicemail = cls(
            action=action,
            message=message,
        )

        session_voicemail.additional_properties = d
        return session_voicemail

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
