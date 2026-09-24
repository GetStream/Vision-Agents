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
    from ..models.knowledge_url_declaration import KnowledgeUrlDeclaration
    from ..models.session_video import SessionVideo
    from ..models.skill_request import SkillRequest
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
            guardrail (str | Unset): The directory's guardrail.md, whole: frontmatter saying how to screen a turn, then the
                policy in prose. Empty means every turn is answered.
            skills (list[SkillRequest] | Unset):
            knowledge (list[KnowledgeDocument] | Unset):
            knowledge_urls (list[KnowledgeUrlDeclaration] | Unset): The pages the directory's knowledge/urls.yaml declares.
                They are subscribed to in the same knowledge base as the files, so one lookup covers both.
            mode (AgentMode | Unset): Whether the agent is spoken to or written to. A voice agent joins a call, transcribes
                what it hears and speaks its replies. A text agent holds the same conversation in writing, so it uses neither
                speech target and a session created from it needs no call to join.
            stt (str | Unset):
            tts (str | Unset):
            sts (str | Unset): A speech-to-speech target: one native audio model that hears the caller and speaks back.
                Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade.
            voice (str | Unset):
            llm (str | Unset):
            video (SessionVideo | Unset):
            subagent (str | Unset):
            search (str | Unset):
            greeting (str | Unset):
            plugins (list[str] | Unset):
            keyterms (list[str] | Unset):
            sandbox (Sandbox | Unset): Where the subagent may run code it writes. Only the subagent is offered it: running
                code takes seconds, and the model holding the conversation has none to spare. Omit it and the subagent works
                everything out in its head.
            tags (SyncAgentRequestTags | Unset):
    """

    name: str
    hash_: str
    instructions: str | Unset = UNSET
    guardrail: str | Unset = UNSET
    skills: list[SkillRequest] | Unset = UNSET
    knowledge: list[KnowledgeDocument] | Unset = UNSET
    knowledge_urls: list[KnowledgeUrlDeclaration] | Unset = UNSET
    mode: AgentMode | Unset = UNSET
    stt: str | Unset = UNSET
    tts: str | Unset = UNSET
    sts: str | Unset = UNSET
    voice: str | Unset = UNSET
    llm: str | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    subagent: str | Unset = UNSET
    search: str | Unset = UNSET
    greeting: str | Unset = UNSET
    plugins: list[str] | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    sandbox: Sandbox | Unset = UNSET
    tags: SyncAgentRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        hash_ = self.hash_

        instructions = self.instructions

        guardrail = self.guardrail

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

        knowledge_urls: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.knowledge_urls, Unset):
            knowledge_urls = []
            for knowledge_urls_item_data in self.knowledge_urls:
                knowledge_urls_item = knowledge_urls_item_data.to_dict()
                knowledge_urls.append(knowledge_urls_item)

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        stt = self.stt

        tts = self.tts

        sts = self.sts

        voice = self.voice

        llm = self.llm

        video: dict[str, Any] | Unset = UNSET
        if not isinstance(self.video, Unset):
            video = self.video.to_dict()

        subagent = self.subagent

        search = self.search

        greeting = self.greeting

        plugins: list[str] | Unset = UNSET
        if not isinstance(self.plugins, Unset):
            plugins = self.plugins

        keyterms: list[str] | Unset = UNSET
        if not isinstance(self.keyterms, Unset):
            keyterms = self.keyterms

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
        if guardrail is not UNSET:
            field_dict["guardrail"] = guardrail
        if skills is not UNSET:
            field_dict["skills"] = skills
        if knowledge is not UNSET:
            field_dict["knowledge"] = knowledge
        if knowledge_urls is not UNSET:
            field_dict["knowledge_urls"] = knowledge_urls
        if mode is not UNSET:
            field_dict["mode"] = mode
        if stt is not UNSET:
            field_dict["stt"] = stt
        if tts is not UNSET:
            field_dict["tts"] = tts
        if sts is not UNSET:
            field_dict["sts"] = sts
        if voice is not UNSET:
            field_dict["voice"] = voice
        if llm is not UNSET:
            field_dict["llm"] = llm
        if video is not UNSET:
            field_dict["video"] = video
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
        if sandbox is not UNSET:
            field_dict["sandbox"] = sandbox
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.knowledge_document import KnowledgeDocument
        from ..models.knowledge_url_declaration import (
            KnowledgeUrlDeclaration,
        )
        from ..models.session_video import SessionVideo
        from ..models.skill_request import SkillRequest
        from ..models.sync_agent_request_tags import (
            SyncAgentRequestTags,
        )

        d = dict(src_dict)
        name = d.pop("name")

        hash_ = d.pop("hash")

        instructions = d.pop("instructions", UNSET)

        guardrail = d.pop("guardrail", UNSET)

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

        _knowledge_urls = d.pop("knowledge_urls", UNSET)
        knowledge_urls: list[KnowledgeUrlDeclaration] | Unset = UNSET
        if _knowledge_urls is not UNSET:
            knowledge_urls = []
            for knowledge_urls_item_data in _knowledge_urls:
                knowledge_urls_item = KnowledgeUrlDeclaration.from_dict(
                    knowledge_urls_item_data
                )

                knowledge_urls.append(knowledge_urls_item)

        _mode = d.pop("mode", UNSET)
        mode: AgentMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = AgentMode(_mode)

        stt = d.pop("stt", UNSET)

        tts = d.pop("tts", UNSET)

        sts = d.pop("sts", UNSET)

        voice = d.pop("voice", UNSET)

        llm = d.pop("llm", UNSET)

        _video = d.pop("video", UNSET)
        video: SessionVideo | Unset
        if isinstance(_video, Unset):
            video = UNSET
        else:
            video = SessionVideo.from_dict(_video)

        subagent = d.pop("subagent", UNSET)

        search = d.pop("search", UNSET)

        greeting = d.pop("greeting", UNSET)

        plugins = cast(list[str], d.pop("plugins", UNSET))

        keyterms = cast(list[str], d.pop("keyterms", UNSET))

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
            guardrail=guardrail,
            skills=skills,
            knowledge=knowledge,
            knowledge_urls=knowledge_urls,
            mode=mode,
            stt=stt,
            tts=tts,
            sts=sts,
            voice=voice,
            llm=llm,
            video=video,
            subagent=subagent,
            search=search,
            greeting=greeting,
            plugins=plugins,
            keyterms=keyterms,
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
