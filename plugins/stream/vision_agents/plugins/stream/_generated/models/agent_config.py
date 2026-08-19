from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_config_tags import AgentConfigTags


T = TypeVar("T", bound="AgentConfig")


@_attrs_define
class AgentConfig:
    """
    Attributes:
        id (str):
        name (str):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        stt (str | Unset):
        tts (str | Unset):
        voice (str | Unset):
        llm (str | Unset):
        subagent (str | Unset):
        instructions (str | Unset):
        greeting (str | Unset):
        skills (list[str] | Unset):
        knowledge_namespace (str | Unset):
        tags (AgentConfigTags | Unset):
    """

    id: str
    name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    stt: str | Unset = UNSET
    tts: str | Unset = UNSET
    voice: str | Unset = UNSET
    llm: str | Unset = UNSET
    subagent: str | Unset = UNSET
    instructions: str | Unset = UNSET
    greeting: str | Unset = UNSET
    skills: list[str] | Unset = UNSET
    knowledge_namespace: str | Unset = UNSET
    tags: AgentConfigTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

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

        knowledge_namespace = self.knowledge_namespace

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "created_at": created_at,
                "updated_at": updated_at,
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
        if knowledge_namespace is not UNSET:
            field_dict["knowledge_namespace"] = knowledge_namespace
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_config_tags import AgentConfigTags

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        stt = d.pop("stt", UNSET)

        tts = d.pop("tts", UNSET)

        voice = d.pop("voice", UNSET)

        llm = d.pop("llm", UNSET)

        subagent = d.pop("subagent", UNSET)

        instructions = d.pop("instructions", UNSET)

        greeting = d.pop("greeting", UNSET)

        skills = cast(list[str], d.pop("skills", UNSET))

        knowledge_namespace = d.pop("knowledge_namespace", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: AgentConfigTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = AgentConfigTags.from_dict(_tags)

        agent_config = cls(
            id=id,
            name=name,
            created_at=created_at,
            updated_at=updated_at,
            stt=stt,
            tts=tts,
            voice=voice,
            llm=llm,
            subagent=subagent,
            instructions=instructions,
            greeting=greeting,
            skills=skills,
            knowledge_namespace=knowledge_namespace,
            tags=tags,
        )

        agent_config.additional_properties = d
        return agent_config

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
