from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.agent_mode import AgentMode
from ..models.sandbox import Sandbox
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.knowledge_document import KnowledgeDocument
    from ..models.session_video import SessionVideo
    from ..models.skill_request import SkillRequest
    from ..models.sync_agent_request_subagents import SyncAgentRequestSubagents
    from ..models.sync_agent_request_tags import SyncAgentRequestTags


T = TypeVar("T", bound="SyncAgentRequest")


@_attrs_define
class SyncAgentRequest:
    """An agent directory as it is on disk. Everything after the knowledge is what the directory's declaration decides
    rather than what it holds, and a setting left out leaves whatever is stored, so a model chosen in the dashboard
    survives a sync that says nothing about it.

        Attributes:
            name (str): What the config is called, which is also the directory's name.
            hash_ (str): A fingerprint of the directory. A second sync with the same hash does nothing.
            instructions (str | Unset):
            skills (list[SkillRequest] | Unset):
            knowledge (list[KnowledgeDocument] | Unset):
            mode (AgentMode | Unset): Whether the agent is spoken to or written to. A voice agent joins a call, transcribes
                what it hears and speaks its replies. A text agent holds the same conversation in writing, so it uses neither
                speech target and a session created from it needs no call to join.
            stt (str | Unset):
            tts (str | Unset):
            voice (str | Unset):
            llm (str | Unset):
            video (SessionVideo | Unset):
            subagents (SyncAgentRequestSubagents | Unset): Named worker targets. Entries merge over stored configuration; an
                empty target removes that worker. Singular subagent is shorthand for default.
            subagent (str | Unset):
            search (str | Unset):
            greeting (str | Unset):
            plugins (list[str] | Unset):
            keyterms (list[str] | Unset):
            sandbox_profile (str | Unset): Backend-managed research profile, scoped to this customer and agent.
            sandbox (Sandbox | Unset): Where the subagent may run code it writes. Only the subagent is offered it: running
                code takes seconds, and the model holding the conversation has none to spare. Omit it and the subagent works
                everything out in its head.
            tags (SyncAgentRequestTags | Unset):
    """

    name: str
    hash_: str
    instructions: str | Unset = UNSET
    skills: list[SkillRequest] | Unset = UNSET
    knowledge: list[KnowledgeDocument] | Unset = UNSET
    mode: AgentMode | Unset = UNSET
    stt: str | Unset = UNSET
    tts: str | Unset = UNSET
    voice: str | Unset = UNSET
    llm: str | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    subagents: SyncAgentRequestSubagents | Unset = UNSET
    subagent: str | Unset = UNSET
    search: str | Unset = UNSET
    greeting: str | Unset = UNSET
    plugins: list[str] | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    sandbox_profile: str | Unset = UNSET
    sandbox: Sandbox | Unset = UNSET
    tags: SyncAgentRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        hash_ = self.hash_

        instructions = self.instructions

        skills: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = []
            for skills_item_data in self.skills:
                skills_item = skills_item_data.to_dict()
                skills.append(skills_item)

        knowledge: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.knowledge, Unset):
            knowledge = []
            for knowledge_item_data in self.knowledge:
                knowledge_item = knowledge_item_data.to_dict()
                knowledge.append(knowledge_item)

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        stt = self.stt

        tts = self.tts

        voice = self.voice

        llm = self.llm

        video: dict[str, Any] | Unset = UNSET
        if not isinstance(self.video, Unset):
            video = self.video.to_dict()

        subagents: dict[str, Any] | Unset = UNSET
        if not isinstance(self.subagents, Unset):
            subagents = self.subagents.to_dict()

        subagent = self.subagent

        search = self.search

        greeting = self.greeting

        plugins: list[str] | Unset = UNSET
        if not isinstance(self.plugins, Unset):
            plugins = self.plugins

        keyterms: list[str] | Unset = UNSET
        if not isinstance(self.keyterms, Unset):
            keyterms = self.keyterms

        sandbox_profile = self.sandbox_profile

        sandbox: str | Unset = UNSET
        if not isinstance(self.sandbox, Unset):
            sandbox = self.sandbox.value

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "hash": hash_,
            }
        )
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if skills is not UNSET:
            field_dict["skills"] = skills
        if knowledge is not UNSET:
            field_dict["knowledge"] = knowledge
        if mode is not UNSET:
            field_dict["mode"] = mode
        if stt is not UNSET:
            field_dict["stt"] = stt
        if tts is not UNSET:
            field_dict["tts"] = tts
        if voice is not UNSET:
            field_dict["voice"] = voice
        if llm is not UNSET:
            field_dict["llm"] = llm
        if video is not UNSET:
            field_dict["video"] = video
        if subagents is not UNSET:
            field_dict["subagents"] = subagents
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if search is not UNSET:
            field_dict["search"] = search
        if greeting is not UNSET:
            field_dict["greeting"] = greeting
        if plugins is not UNSET:
            field_dict["plugins"] = plugins
        if keyterms is not UNSET:
            field_dict["keyterms"] = keyterms
        if sandbox_profile is not UNSET:
            field_dict["sandbox_profile"] = sandbox_profile
        if sandbox is not UNSET:
            field_dict["sandbox"] = sandbox
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.knowledge_document import KnowledgeDocument
        from ..models.session_video import SessionVideo
        from ..models.skill_request import SkillRequest
        from ..models.sync_agent_request_subagents import (
            SyncAgentRequestSubagents,
        )
        from ..models.sync_agent_request_tags import (
            SyncAgentRequestTags,
        )

        d = dict(src_dict)
        name = d.pop("name")

        hash_ = d.pop("hash")

        instructions = d.pop("instructions", UNSET)

        _skills = d.pop("skills", UNSET)
        skills: list[SkillRequest] | Unset = UNSET
        if _skills is not UNSET:
            skills = []
            for skills_item_data in _skills:
                skills_item = SkillRequest.from_dict(skills_item_data)

                skills.append(skills_item)

        _knowledge = d.pop("knowledge", UNSET)
        knowledge: list[KnowledgeDocument] | Unset = UNSET
        if _knowledge is not UNSET:
            knowledge = []
            for knowledge_item_data in _knowledge:
                knowledge_item = KnowledgeDocument.from_dict(knowledge_item_data)

                knowledge.append(knowledge_item)

        _mode = d.pop("mode", UNSET)
        mode: AgentMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = AgentMode(_mode)

        stt = d.pop("stt", UNSET)

        tts = d.pop("tts", UNSET)

        voice = d.pop("voice", UNSET)

        llm = d.pop("llm", UNSET)

        _video = d.pop("video", UNSET)
        video: SessionVideo | Unset
        if isinstance(_video, Unset):
            video = UNSET
        else:
            video = SessionVideo.from_dict(_video)

        _subagents = d.pop("subagents", UNSET)
        subagents: SyncAgentRequestSubagents | Unset
        if isinstance(_subagents, Unset):
            subagents = UNSET
        else:
            subagents = SyncAgentRequestSubagents.from_dict(_subagents)

        subagent = d.pop("subagent", UNSET)

        search = d.pop("search", UNSET)

        greeting = d.pop("greeting", UNSET)

        plugins = cast(list[str], d.pop("plugins", UNSET))

        keyterms = cast(list[str], d.pop("keyterms", UNSET))

        sandbox_profile = d.pop("sandbox_profile", UNSET)

        _sandbox = d.pop("sandbox", UNSET)
        sandbox: Sandbox | Unset
        if isinstance(_sandbox, Unset):
            sandbox = UNSET
        else:
            sandbox = Sandbox(_sandbox)

        _tags = d.pop("tags", UNSET)
        tags: SyncAgentRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = SyncAgentRequestTags.from_dict(_tags)

        sync_agent_request = cls(
            name=name,
            hash_=hash_,
            instructions=instructions,
            skills=skills,
            knowledge=knowledge,
            mode=mode,
            stt=stt,
            tts=tts,
            voice=voice,
            llm=llm,
            video=video,
            subagents=subagents,
            subagent=subagent,
            search=search,
            greeting=greeting,
            plugins=plugins,
            keyterms=keyterms,
            sandbox_profile=sandbox_profile,
            sandbox=sandbox,
            tags=tags,
        )

        sync_agent_request.additional_properties = d
        return sync_agent_request

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
