from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SessionToolApproval")


@_attrs_define
class SessionToolApproval:
    """Says a person must allow each call before it runs. In a persistent conversation the call's ai_tool_call attachment
    opens as awaiting_approval, addressed to the person whose command it answers (and, for a client tool, their
    install), and carries this question for their client to ask. The caller collects the answer and reports it over the
    events socket with tool_approval: allowed, the call goes on as it would have (awaiting_client for a client tool,
    running otherwise); declined, it is cancelled. The caller still answers the call with tool_result either way. Every
    channel member can read the question.

        Attributes:
            title (str): The question, such as "Share your location?".
            allow_title (str | Unset): The label of the button that allows the call.
            decline_title (str | Unset): The label of the button that declines it.
            message (str | Unset): What allowing it shares or does, such as "Only your city is shared."
            reason_argument (str | Unset): The argument, a string, in which the model says why it wants this call. Its text
                (at most 160 characters) is shown as the approval's reason, so it is visible to every channel member even for a
                server tool.
    """

    title: str
    allow_title: str | Unset = UNSET
    decline_title: str | Unset = UNSET
    message: str | Unset = UNSET
    reason_argument: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        title = self.title

        allow_title = self.allow_title

        decline_title = self.decline_title

        message = self.message

        reason_argument = self.reason_argument

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "title": title,
            }
        )
        if allow_title is not UNSET:
            field_dict["allow_title"] = allow_title
        if decline_title is not UNSET:
            field_dict["decline_title"] = decline_title
        if message is not UNSET:
            field_dict["message"] = message
        if reason_argument is not UNSET:
            field_dict["reason_argument"] = reason_argument

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        title = d.pop("title")

        allow_title = d.pop("allow_title", UNSET)

        decline_title = d.pop("decline_title", UNSET)

        message = d.pop("message", UNSET)

        reason_argument = d.pop("reason_argument", UNSET)

        session_tool_approval = cls(
            title=title,
            allow_title=allow_title,
            decline_title=decline_title,
            message=message,
            reason_argument=reason_argument,
        )

        session_tool_approval.additional_properties = d
        return session_tool_approval

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
