from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.session_state import SessionState
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_subagents import SessionSubagents
    from ..models.session_video import SessionVideo


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
        conversation_id (str | Unset): Stream Chat CID to resume; returned for persistent text sessions.
        persist_conversation (bool | Unset): Persist a text conversation in Stream Chat, creating a channel when no CID
            is supplied.
        context_truncated (bool | Unset): Older history was omitted from the model context.
        text (bool | Unset): The conversation is held in writing rather than on a call.
        llm (str | Unset): The provider and model answering, once routing has picked one.
        tts (str | Unset): The provider and model speaking.
        stt (str | Unset): The provider and model transcribing, once somebody has been heard.
        subagent (str | Unset): The provider and model delegated work runs on.
        subagents (SessionSubagents | Unset): Configured worker targets, prepared asynchronously.
        video (SessionVideo | Unset):
        instructions (str | Unset):
    """

    id: str
    call_id: str
    call_type: str
    user_id: str
    agent_id: str
    state: SessionState
    created_at: datetime.datetime
    conversation_id: str | Unset = UNSET
    persist_conversation: bool | Unset = UNSET
    context_truncated: bool | Unset = UNSET
    text: bool | Unset = UNSET
    llm: str | Unset = UNSET
    tts: str | Unset = UNSET
    stt: str | Unset = UNSET
    subagent: str | Unset = UNSET
    subagents: SessionSubagents | Unset = UNSET
    video: SessionVideo | Unset = UNSET
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

        conversation_id = self.conversation_id

        persist_conversation = self.persist_conversation

        context_truncated = self.context_truncated

        text = self.text

        llm = self.llm

        tts = self.tts

        stt = self.stt

        subagent = self.subagent

        subagents: dict[str, Any] | Unset = UNSET
        if not isinstance(self.subagents, Unset):
            subagents = self.subagents.to_dict()

        video: dict[str, Any] | Unset = UNSET
        if not isinstance(self.video, Unset):
            video = self.video.to_dict()

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
        if conversation_id is not UNSET:
            field_dict["conversation_id"] = conversation_id
        if persist_conversation is not UNSET:
            field_dict["persist_conversation"] = persist_conversation
        if context_truncated is not UNSET:
            field_dict["context_truncated"] = context_truncated
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
        if subagents is not UNSET:
            field_dict["subagents"] = subagents
        if video is not UNSET:
            field_dict["video"] = video
        if instructions is not UNSET:
            field_dict["instructions"] = instructions

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.session_subagents import SessionSubagents
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
        id = d.pop("id")

        call_id = d.pop("call_id")

        call_type = d.pop("call_type")

        user_id = d.pop("user_id")

        agent_id = d.pop("agent_id")

        state = SessionState(d.pop("state"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        conversation_id = d.pop("conversation_id", UNSET)

        persist_conversation = d.pop("persist_conversation", UNSET)

        context_truncated = d.pop("context_truncated", UNSET)

        text = d.pop("text", UNSET)

        llm = d.pop("llm", UNSET)

        tts = d.pop("tts", UNSET)

        stt = d.pop("stt", UNSET)

        subagent = d.pop("subagent", UNSET)

        _subagents = d.pop("subagents", UNSET)
        subagents: SessionSubagents | Unset
        if isinstance(_subagents, Unset):
            subagents = UNSET
        else:
            subagents = SessionSubagents.from_dict(_subagents)

        _video = d.pop("video", UNSET)
        video: SessionVideo | Unset
        if isinstance(_video, Unset):
            video = UNSET
        else:
            video = SessionVideo.from_dict(_video)

        instructions = d.pop("instructions", UNSET)

        session = cls(
            id=id,
            call_id=call_id,
            call_type=call_type,
            user_id=user_id,
            agent_id=agent_id,
            state=state,
            created_at=created_at,
            conversation_id=conversation_id,
            persist_conversation=persist_conversation,
            context_truncated=context_truncated,
            text=text,
            llm=llm,
            tts=tts,
            stt=stt,
            subagent=subagent,
            subagents=subagents,
            video=video,
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
