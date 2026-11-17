from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.session_end_reason import SessionEndReason
from ..types import UNSET, Unset

T = TypeVar("T", bound="ActivityEnding")


@_attrs_define
class ActivityEnding:
    """
    Attributes:
        sessions (int): How many ended so.
        end_reason (SessionEndReason | Unset): Why the session ended, with whose fault it was and the stage it came at:
            agent_hangup, the agent hung up (fault none, stage conversation); transferred, the agent handed the caller to a
            person (fault none, stage conversation); hangup, the caller left (fault caller, stage conversation); start,
            nobody joined the phone call in time (fault carrier, stage ringing); unanswered, the call placed was never
            answered (fault caller, stage ringing); voicemail, an answering machine picked up the call placed, and the agent
            left its message or hung up (fault none, stage ringing); closed, the session was closed through the API, or
            replaced by a newer one (fault none, stage conversation); shutdown, the router ended the session as it shut down
            (fault router, stage conversation); agent_failed, the agent left the call or answered no turn for a failure of
            the router's own (fault router, stage conversation); model_failed, a model vendor failed: the agent left the
            call or answered no turn (fault vendor, stage conversation); model_config_refused, the model failed on the
            customer's own key or model, or refused a setting, and the agent answered no turn (fault customer_config, stage
            conversation); model_no_credit, the credit or spend limits refused the model's work, and the agent answered no
            turn (fault customer_credit, stage conversation).
    """

    sessions: int
    end_reason: SessionEndReason | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sessions = self.sessions

        end_reason: str | Unset = UNSET
        if not isinstance(self.end_reason, Unset):
            end_reason = self.end_reason.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "sessions": sessions,
            }
        )
        if end_reason is not UNSET:
            field_dict["end_reason"] = end_reason

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        sessions = d.pop("sessions")

        _end_reason = d.pop("end_reason", UNSET)
        end_reason: SessionEndReason | Unset
        if isinstance(_end_reason, Unset):
            end_reason = UNSET
        else:
            end_reason = SessionEndReason._decode(_end_reason)

        activity_ending = cls(
            sessions=sessions,
            end_reason=end_reason,
        )

        activity_ending.additional_properties = d
        return activity_ending

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
