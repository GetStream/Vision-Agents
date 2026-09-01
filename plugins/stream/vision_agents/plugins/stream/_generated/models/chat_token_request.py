from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ChatTokenRequest")


@_attrs_define
class ChatTokenRequest:
    """
    Attributes:
        agent_id (str): Whose conversation to read. This is the session's agent id, which is what names the channel it
            is written to.
        user_id (str | Unset): Who the browser reads as. Somebody watching is not the agent, so this defaults to a
            reader of its own rather than to the agent's user.
        user_name (str | Unset): The name shown against anything they write. Defaults to the user id.
    """

    agent_id: str
    user_id: str | Unset = UNSET
    user_name: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent_id = self.agent_id

        user_id = self.user_id

        user_name = self.user_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "agent_id": agent_id,
            }
        )
        if user_id is not UNSET:
            field_dict["user_id"] = user_id
        if user_name is not UNSET:
            field_dict["user_name"] = user_name

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        agent_id = d.pop("agent_id")

        user_id = d.pop("user_id", UNSET)

        user_name = d.pop("user_name", UNSET)

        chat_token_request = cls(
            agent_id=agent_id,
            user_id=user_id,
            user_name=user_name,
        )

        chat_token_request.additional_properties = d
        return chat_token_request

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
