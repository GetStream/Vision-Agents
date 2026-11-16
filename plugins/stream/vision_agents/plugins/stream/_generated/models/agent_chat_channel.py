from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.chat_channel_answer import ChatChannelAnswer
from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentChatChannel")


@_attrs_define
class AgentChatChannel:
    """
    Attributes:
        type_ (str): The channel type's name, as it comes before the colon in a channel's cid. Example: callouts.
        answer (ChatChannelAnswer | Unset): Which messages in a channel of the type are answered. always answers every
            message. mentioned answers only a message that mentions the agent: its mentioned_users holds the Stream user
            whose id is the config's id, or its text has @ and that user's id or the config's name, as Stream's clients
            write a mention. A message that does not mention the agent gets no reply and starts no model; it stays in the
            channel, and the next answered message reads it as what was said before. Omitted is always.
    """

    type_: str
    answer: ChatChannelAnswer | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        answer: str | Unset = UNSET
        if not isinstance(self.answer, Unset):
            answer = self.answer.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
            }
        )
        if answer is not UNSET:
            field_dict["answer"] = answer

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        type_ = d.pop("type")

        _answer = d.pop("answer", UNSET)
        answer: ChatChannelAnswer | Unset
        if isinstance(_answer, Unset):
            answer = UNSET
        else:
            answer = ChatChannelAnswer._decode(_answer)

        agent_chat_channel = cls(
            type_=type_,
            answer=answer,
        )

        agent_chat_channel.additional_properties = d
        return agent_chat_channel

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
