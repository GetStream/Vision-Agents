from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.session_state import SessionState
from ..types import UNSET, Unset

T = TypeVar("T", bound="Session")


@_attrs_define
class Session:
    """
    Attributes:
        id (str):
        call_id (str): Empty for a text session, which joins no call.
        call_type (str):
        user_id (str):
        agent_id (str):
        state (SessionState): Whether the agent is still in the call.
        created_at (datetime.datetime):
        text (bool | Unset): The conversation is held in writing rather than on a call.
        llm (str | Unset): The provider and model answering, once routing has picked one.
        tts (str | Unset): The provider and model speaking.
        stt (str | Unset): The provider and model transcribing, once somebody has been heard.
        subagent (str | Unset): The provider and model delegated work runs on.
        instructions (str | Unset):
    """

    id: str
    call_id: str
    call_type: str
    user_id: str
    agent_id: str
    state: SessionState
    created_at: datetime.datetime
    text: bool | Unset = UNSET
    llm: str | Unset = UNSET
    tts: str | Unset = UNSET
    stt: str | Unset = UNSET
    subagent: str | Unset = UNSET
    instructions: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        call_id = self.call_id

        call_type = self.call_type

        user_id = self.user_id

        agent_id = self.agent_id

        state = self.state.value

        created_at = self.created_at.isoformat()

        text = self.text

        llm = self.llm

        tts = self.tts

        stt = self.stt

        subagent = self.subagent

        instructions = self.instructions

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "call_id": call_id,
                "call_type": call_type,
                "user_id": user_id,
                "agent_id": agent_id,
                "state": state,
                "created_at": created_at,
            }
        )
        if text is not UNSET:
            field_dict["text"] = text
        if llm is not UNSET:
            field_dict["llm"] = llm
        if tts is not UNSET:
            field_dict["tts"] = tts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if instructions is not UNSET:
            field_dict["instructions"] = instructions

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        id = d.pop("id")

        call_id = d.pop("call_id")

        call_type = d.pop("call_type")

        user_id = d.pop("user_id")

        agent_id = d.pop("agent_id")

        state = SessionState(d.pop("state"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        text = d.pop("text", UNSET)

        llm = d.pop("llm", UNSET)

        tts = d.pop("tts", UNSET)

        stt = d.pop("stt", UNSET)

        subagent = d.pop("subagent", UNSET)

        instructions = d.pop("instructions", UNSET)

        session = cls(
            id=id,
            call_id=call_id,
            call_type=call_type,
            user_id=user_id,
            agent_id=agent_id,
            state=state,
            created_at=created_at,
            text=text,
            llm=llm,
            tts=tts,
            stt=stt,
            subagent=subagent,
            instructions=instructions,
        )

        session.additional_properties = d
        return session

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
