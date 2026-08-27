from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_config_request_tags import AgentConfigRequestTags


T = TypeVar("T", bound="AgentConfigRequest")


@_attrs_define
class AgentConfigRequest:
    """
    Attributes:
        name (str): What the config is called, which is unique among the customer's own.
        stt (str | Unset): A provider/model or a capability shortcut. Empty leaves the default.
        tts (str | Unset):
        voice (str | Unset): Provider-specific voice id.
        llm (str | Unset): The model holding the conversation.
        subagent (str | Unset): The model that does the thinking. Empty means the voice model answers everything itself,
            and skills mean nothing.
        instructions (str | Unset):
        greeting (str | Unset):
        skills (list[str] | Unset): Skill names, either the customer's own or one of the built-in think, recall and
            explain. Omit for the built-in set.
        keyterms (list[str] | Unset): Business-specific words the transcriber would otherwise get wrong, such as product
            or company names. Up to 100 terms, and providers that cannot be told about vocabulary ignore them.
        knowledge_namespace (str | Unset): What the agent may look things up in. Empty means it knows only what it was
            told.
        tags (AgentConfigRequestTags | Unset): Cost labels, carried onto every request a session using it makes.
    """

    name: str
    stt: str | Unset = UNSET
    tts: str | Unset = UNSET
    voice: str | Unset = UNSET
    llm: str | Unset = UNSET
    subagent: str | Unset = UNSET
    instructions: str | Unset = UNSET
    greeting: str | Unset = UNSET
    skills: list[str] | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    knowledge_namespace: str | Unset = UNSET
    tags: AgentConfigRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        stt = self.stt

        tts = self.tts

        voice = self.voice

        llm = self.llm

        subagent = self.subagent

        instructions = self.instructions

        greeting = self.greeting

        skills: list[str] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = self.skills

        keyterms: list[str] | Unset = UNSET
        if not isinstance(self.keyterms, Unset):
            keyterms = self.keyterms

        knowledge_namespace = self.knowledge_namespace

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
            }
        )
        if stt is not UNSET:
            field_dict["stt"] = stt
        if tts is not UNSET:
            field_dict["tts"] = tts
        if voice is not UNSET:
            field_dict["voice"] = voice
        if llm is not UNSET:
            field_dict["llm"] = llm
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if greeting is not UNSET:
            field_dict["greeting"] = greeting
        if skills is not UNSET:
            field_dict["skills"] = skills
        if keyterms is not UNSET:
            field_dict["keyterms"] = keyterms
        if knowledge_namespace is not UNSET:
            field_dict["knowledge_namespace"] = knowledge_namespace
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_config_request_tags import AgentConfigRequestTags

        d = dict(src_dict)
        name = d.pop("name")

        stt = d.pop("stt", UNSET)

        tts = d.pop("tts", UNSET)

        voice = d.pop("voice", UNSET)

        llm = d.pop("llm", UNSET)

        subagent = d.pop("subagent", UNSET)

        instructions = d.pop("instructions", UNSET)

        greeting = d.pop("greeting", UNSET)

        skills = cast(list[str], d.pop("skills", UNSET))

        keyterms = cast(list[str], d.pop("keyterms", UNSET))

        knowledge_namespace = d.pop("knowledge_namespace", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: AgentConfigRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = AgentConfigRequestTags.from_dict(_tags)

        agent_config_request = cls(
            name=name,
            stt=stt,
            tts=tts,
            voice=voice,
            llm=llm,
            subagent=subagent,
            instructions=instructions,
            greeting=greeting,
            skills=skills,
            keyterms=keyterms,
            knowledge_namespace=knowledge_namespace,
            tags=tags,
        )

        agent_config_request.additional_properties = d
        return agent_config_request

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
