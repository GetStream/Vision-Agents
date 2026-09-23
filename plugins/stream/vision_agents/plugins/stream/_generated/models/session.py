from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.session_mode import SessionMode
from ..models.session_state import SessionState
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_overwrites import ModelOverwrites
    from ..models.session_custom import SessionCustom
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
        sts (str | Unset): The provider and model holding a native conversation, once routing has picked one.
        stt (str | Unset): The provider and model transcribing, once somebody has been heard.
        subagent (str | Unset): The provider and model delegated work runs on.
        voice (str | Unset): The voice speaking, in the provider's own terms. It is the provider's default when the
            session asked for none.
        mode (SessionMode | Unset): How the session hears and speaks: a transcriber, a conversation model and a voice;
            one speech-to-speech model; or in writing.
        video (SessionVideo | Unset):
        instructions (str | Unset):
        agent (str | Unset): The name the agent was addressed as. Recorded on the session as well as the config id, so
            renaming a config does not rewrite what older sessions were opened against.
        config_id (str | Unset): The agent config the session ran under, empty for one that spelled itself out.
        incognito (bool | Unset): Nothing about this session was recorded. It is reported so a caller can see that what
            they asked for is what they got, but it is never read back from storage: an incognito session has no row to read
            it from.
        title (str | Unset):
        description (str | Unset):
        project (str | Unset):
        custom (SessionCustom | Unset):
        model_overwrites (ModelOverwrites | Unset): What to change about the models for one session, over whatever its
            agent config decided.
            It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides
            the config, and a caller reading a session back wants to see what they changed in one place rather than diffed
            against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because
            a caller able to rewrite those could make a session impersonate a different agent.
        forked_from (str | Unset): The session this one continued from, empty for one opened fresh.
        closed_at (datetime.datetime | Unset): When the session ended. Absent while it is still running.
        last_response_at (datetime.datetime | Unset): When the agent last answered. This is what a most-recently-used
            ordering of conversations reads, since a session renamed long after it ended has not become more recent.
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
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    subagent: str | Unset = UNSET
    voice: str | Unset = UNSET
    mode: SessionMode | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    instructions: str | Unset = UNSET
    agent: str | Unset = UNSET
    config_id: str | Unset = UNSET
    incognito: bool | Unset = UNSET
    title: str | Unset = UNSET
    description: str | Unset = UNSET
    project: str | Unset = UNSET
    custom: SessionCustom | Unset = UNSET
    model_overwrites: ModelOverwrites | Unset = UNSET
    forked_from: str | Unset = UNSET
    closed_at: datetime.datetime | Unset = UNSET
    last_response_at: datetime.datetime | Unset = UNSET
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

        sts = self.sts

        stt = self.stt

        subagent = self.subagent

        voice = self.voice

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        video: dict[str, Any] | Unset = UNSET
        if not isinstance(self.video, Unset):
            video = self.video.to_dict()

        instructions = self.instructions

        agent = self.agent

        config_id = self.config_id

        incognito = self.incognito

        title = self.title

        description = self.description

        project = self.project

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        model_overwrites: dict[str, Any] | Unset = UNSET
        if not isinstance(self.model_overwrites, Unset):
            model_overwrites = self.model_overwrites.to_dict()

        forked_from = self.forked_from

        closed_at: str | Unset = UNSET
        if not isinstance(self.closed_at, Unset):
            closed_at = self.closed_at.isoformat()

        last_response_at: str | Unset = UNSET
        if not isinstance(self.last_response_at, Unset):
            last_response_at = self.last_response_at.isoformat()

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
        if sts is not UNSET:
            field_dict["sts"] = sts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if voice is not UNSET:
            field_dict["voice"] = voice
        if mode is not UNSET:
            field_dict["mode"] = mode
        if video is not UNSET:
            field_dict["video"] = video
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if agent is not UNSET:
            field_dict["agent"] = agent
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if incognito is not UNSET:
            field_dict["incognito"] = incognito
        if title is not UNSET:
            field_dict["title"] = title
        if description is not UNSET:
            field_dict["description"] = description
        if project is not UNSET:
            field_dict["project"] = project
        if custom is not UNSET:
            field_dict["custom"] = custom
        if model_overwrites is not UNSET:
            field_dict["model_overwrites"] = model_overwrites
        if forked_from is not UNSET:
            field_dict["forked_from"] = forked_from
        if closed_at is not UNSET:
            field_dict["closed_at"] = closed_at
        if last_response_at is not UNSET:
            field_dict["last_response_at"] = last_response_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.model_overwrites import ModelOverwrites
        from ..models.session_custom import SessionCustom
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

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        subagent = d.pop("subagent", UNSET)

        voice = d.pop("voice", UNSET)

        _mode = d.pop("mode", UNSET)
        mode: SessionMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = SessionMode(_mode)

        _video = d.pop("video", UNSET)
        video: SessionVideo | Unset
        if isinstance(_video, Unset):
            video = UNSET
        else:
            video = SessionVideo.from_dict(_video)

        instructions = d.pop("instructions", UNSET)

        agent = d.pop("agent", UNSET)

        config_id = d.pop("config_id", UNSET)

        incognito = d.pop("incognito", UNSET)

        title = d.pop("title", UNSET)

        description = d.pop("description", UNSET)

        project = d.pop("project", UNSET)

        _custom = d.pop("custom", UNSET)
        custom: SessionCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = SessionCustom.from_dict(_custom)

        _model_overwrites = d.pop("model_overwrites", UNSET)
        model_overwrites: ModelOverwrites | Unset
        if isinstance(_model_overwrites, Unset):
            model_overwrites = UNSET
        else:
            model_overwrites = ModelOverwrites.from_dict(_model_overwrites)

        forked_from = d.pop("forked_from", UNSET)

        _closed_at = d.pop("closed_at", UNSET)
        closed_at: datetime.datetime | Unset
        if isinstance(_closed_at, Unset):
            closed_at = UNSET
        else:
            closed_at = datetime.datetime.fromisoformat(_closed_at)

        _last_response_at = d.pop("last_response_at", UNSET)
        last_response_at: datetime.datetime | Unset
        if isinstance(_last_response_at, Unset):
            last_response_at = UNSET
        else:
            last_response_at = datetime.datetime.fromisoformat(_last_response_at)

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
            sts=sts,
            stt=stt,
            subagent=subagent,
            voice=voice,
            mode=mode,
            video=video,
            instructions=instructions,
            agent=agent,
            config_id=config_id,
            incognito=incognito,
            title=title,
            description=description,
            project=project,
            custom=custom,
            model_overwrites=model_overwrites,
            forked_from=forked_from,
            closed_at=closed_at,
            last_response_at=last_response_at,
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
