from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.session_modality import SessionModality
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
        agent_id (str):
        call_id (str): Empty for a text session, which joins no call.
        call_type (str):
        created_at (datetime.datetime):
        id (str):
        modality (SessionModality): How the user took part: text for a conversation held in writing, voice for a call,
            and video once the agent has seen the user's video. It only moves up, from text or voice to video.
        state (SessionState): Whether the agent is still in the call.
        user_id (str):
        agent (str | Unset): The name the agent was addressed as. Recorded on the session as well as the config id, so
            renaming a config does not rewrite what older sessions were opened against.
        closed_at (datetime.datetime | Unset): When the session ended. Absent while it is still running.
        config_id (str | Unset): The agent config the session ran under, empty for one that spelled itself out.
        context_truncated (bool | Unset): Older history was omitted from the model context.
        conversation_id (str | Unset): Stream Chat CID to resume; returned for persistent text sessions.
        custom (SessionCustom | Unset):
        description (str | Unset):
        forked_from (str | Unset): The session this one continued from, empty for one opened fresh.
        incognito (bool | Unset): Nothing about this session was recorded. It is reported so a caller can see that what
            they asked for is what they got, but it is never read back from storage: an incognito session has no row to read
            it from.
        instructions (str | Unset):
        last_response_at (datetime.datetime | Unset): When the agent last answered. This is what a most-recently-used
            ordering of conversations reads, since a session renamed long after it ended has not become more recent.
        llm (str | Unset): The provider and model answering, once routing has picked one.
        mode (SessionMode | Unset): How the session hears and speaks: a transcriber, a conversation model and a voice;
            one speech-to-speech model; or in writing.
        model_overwrites (ModelOverwrites | Unset): What to change about the models for one session, over whatever its
            agent config decided.
            It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides
            the config, and a caller reading a session back wants to see what they changed in one place rather than diffed
            against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because
            a caller able to rewrite those could make a session impersonate a different agent.
        project_id (str | Unset):
        sts (str | Unset): The provider and model holding a native conversation, once routing has picked one.
        stt (str | Unset): The provider and model transcribing, once somebody has been heard.
        subagent (str | Unset): The provider and model delegated work runs on.
        text (bool | Unset): The conversation is held in writing rather than on a call.
        title (str | Unset):
        tts (str | Unset): The provider and model speaking.
        video (SessionVideo | Unset):
        voice (str | Unset): The voice speaking, in the provider's own terms. It is the provider's default when the
            session asked for none.
    """

    agent_id: str
    call_id: str
    call_type: str
    created_at: datetime.datetime
    id: str
    modality: SessionModality
    state: SessionState
    user_id: str
    agent: str | Unset = UNSET
    closed_at: datetime.datetime | Unset = UNSET
    config_id: str | Unset = UNSET
    context_truncated: bool | Unset = UNSET
    conversation_id: str | Unset = UNSET
    custom: SessionCustom | Unset = UNSET
    description: str | Unset = UNSET
    forked_from: str | Unset = UNSET
    incognito: bool | Unset = UNSET
    instructions: str | Unset = UNSET
    last_response_at: datetime.datetime | Unset = UNSET
    llm: str | Unset = UNSET
    mode: SessionMode | Unset = UNSET
    model_overwrites: ModelOverwrites | Unset = UNSET
    project_id: str | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    subagent: str | Unset = UNSET
    text: bool | Unset = UNSET
    title: str | Unset = UNSET
    tts: str | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    voice: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent_id = self.agent_id

        call_id = self.call_id

        call_type = self.call_type

        created_at = self.created_at.isoformat()

        id = self.id

        modality = self.modality.value

        state = self.state.value

        user_id = self.user_id

        agent = self.agent

        closed_at: str | Unset = UNSET
        if not isinstance(self.closed_at, Unset):
            closed_at = self.closed_at.isoformat()

        config_id = self.config_id

        context_truncated = self.context_truncated

        conversation_id = self.conversation_id

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        description = self.description

        forked_from = self.forked_from

        incognito = self.incognito

        instructions = self.instructions

        last_response_at: str | Unset = UNSET
        if not isinstance(self.last_response_at, Unset):
            last_response_at = self.last_response_at.isoformat()

        llm = self.llm

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        model_overwrites: dict[str, Any] | Unset = UNSET
        if not isinstance(self.model_overwrites, Unset):
            model_overwrites = self.model_overwrites.to_dict()

        project_id = self.project_id

        sts = self.sts

        stt = self.stt

        subagent = self.subagent

        text = self.text

        title = self.title

        tts = self.tts

        video: dict[str, Any] | Unset = UNSET
        if not isinstance(self.video, Unset):
            video = self.video.to_dict()

        voice = self.voice

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "agent_id": agent_id,
                "call_id": call_id,
                "call_type": call_type,
                "created_at": created_at,
                "id": id,
                "modality": modality,
                "state": state,
                "user_id": user_id,
            }
        )
        if agent is not UNSET:
            field_dict["agent"] = agent
        if closed_at is not UNSET:
            field_dict["closed_at"] = closed_at
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if context_truncated is not UNSET:
            field_dict["context_truncated"] = context_truncated
        if conversation_id is not UNSET:
            field_dict["conversation_id"] = conversation_id
        if custom is not UNSET:
            field_dict["custom"] = custom
        if description is not UNSET:
            field_dict["description"] = description
        if forked_from is not UNSET:
            field_dict["forked_from"] = forked_from
        if incognito is not UNSET:
            field_dict["incognito"] = incognito
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if last_response_at is not UNSET:
            field_dict["last_response_at"] = last_response_at
        if llm is not UNSET:
            field_dict["llm"] = llm
        if mode is not UNSET:
            field_dict["mode"] = mode
        if model_overwrites is not UNSET:
            field_dict["model_overwrites"] = model_overwrites
        if project_id is not UNSET:
            field_dict["project_id"] = project_id
        if sts is not UNSET:
            field_dict["sts"] = sts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if text is not UNSET:
            field_dict["text"] = text
        if title is not UNSET:
            field_dict["title"] = title
        if tts is not UNSET:
            field_dict["tts"] = tts
        if video is not UNSET:
            field_dict["video"] = video
        if voice is not UNSET:
            field_dict["voice"] = voice

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.model_overwrites import ModelOverwrites
        from ..models.session_custom import SessionCustom
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
        agent_id = d.pop("agent_id")

        call_id = d.pop("call_id")

        call_type = d.pop("call_type")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        modality = SessionModality(d.pop("modality"))

        state = SessionState(d.pop("state"))

        user_id = d.pop("user_id")

        agent = d.pop("agent", UNSET)

        _closed_at = d.pop("closed_at", UNSET)
        closed_at: datetime.datetime | Unset
        if isinstance(_closed_at, Unset):
            closed_at = UNSET
        else:
            closed_at = datetime.datetime.fromisoformat(_closed_at)

        config_id = d.pop("config_id", UNSET)

        context_truncated = d.pop("context_truncated", UNSET)

        conversation_id = d.pop("conversation_id", UNSET)

        _custom = d.pop("custom", UNSET)
        custom: SessionCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = SessionCustom.from_dict(_custom)

        description = d.pop("description", UNSET)

        forked_from = d.pop("forked_from", UNSET)

        incognito = d.pop("incognito", UNSET)

        instructions = d.pop("instructions", UNSET)

        _last_response_at = d.pop("last_response_at", UNSET)
        last_response_at: datetime.datetime | Unset
        if isinstance(_last_response_at, Unset):
            last_response_at = UNSET
        else:
            last_response_at = datetime.datetime.fromisoformat(_last_response_at)

        llm = d.pop("llm", UNSET)

        _mode = d.pop("mode", UNSET)
        mode: SessionMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = SessionMode(_mode)

        _model_overwrites = d.pop("model_overwrites", UNSET)
        model_overwrites: ModelOverwrites | Unset
        if isinstance(_model_overwrites, Unset):
            model_overwrites = UNSET
        else:
            model_overwrites = ModelOverwrites.from_dict(_model_overwrites)

        project_id = d.pop("project_id", UNSET)

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        subagent = d.pop("subagent", UNSET)

        text = d.pop("text", UNSET)

        title = d.pop("title", UNSET)

        tts = d.pop("tts", UNSET)

        _video = d.pop("video", UNSET)
        video: SessionVideo | Unset
        if isinstance(_video, Unset):
            video = UNSET
        else:
            video = SessionVideo.from_dict(_video)

        voice = d.pop("voice", UNSET)

        session = cls(
            agent_id=agent_id,
            call_id=call_id,
            call_type=call_type,
            created_at=created_at,
            id=id,
            modality=modality,
            state=state,
            user_id=user_id,
            agent=agent,
            closed_at=closed_at,
            config_id=config_id,
            context_truncated=context_truncated,
            conversation_id=conversation_id,
            custom=custom,
            description=description,
            forked_from=forked_from,
            incognito=incognito,
            instructions=instructions,
            last_response_at=last_response_at,
            llm=llm,
            mode=mode,
            model_overwrites=model_overwrites,
            project_id=project_id,
            sts=sts,
            stt=stt,
            subagent=subagent,
            text=text,
            title=title,
            tts=tts,
            video=video,
            voice=voice,
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
