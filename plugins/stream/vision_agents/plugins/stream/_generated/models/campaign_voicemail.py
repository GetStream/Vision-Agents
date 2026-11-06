from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.campaign_voicemail_action import CampaignVoicemailAction
from ..types import UNSET, Unset

T = TypeVar("T", bound="CampaignVoicemail")


@_attrs_define
class CampaignVoicemail:
    """
    Attributes:
        action (CampaignVoicemailAction): off: the agent talks to a machine as to a person. The default.
            hang_up: end the call at once, saying nothing.
            leave_message: say message once the greeting has ended or once the call counts as a machine, then hang up.
        message (str | Unset): {{variable}} placeholders are filled from the contact, then it is said word for word once
            the greeting has ended, and the call ends. Required to launch for leave_message, and not only spaces.
    """

    action: CampaignVoicemailAction
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
        action = CampaignVoicemailAction._decode(d.pop("action"))

        message = d.pop("message", UNSET)

        campaign_voicemail = cls(
            action=action,
            message=message,
        )

        campaign_voicemail.additional_properties = d
        return campaign_voicemail

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
