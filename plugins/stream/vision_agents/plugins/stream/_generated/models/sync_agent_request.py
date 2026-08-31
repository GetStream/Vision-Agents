from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.knowledge_document import KnowledgeDocument
    from ..models.skill_request import SkillRequest


T = TypeVar("T", bound="SyncAgentRequest")


@_attrs_define
class SyncAgentRequest:
    """
    Attributes:
        name (str): What the config is called, which is also the directory's name.
        hash_ (str): A fingerprint of the directory. A second sync with the same hash does nothing.
        instructions (str | Unset):
        skills (list[SkillRequest] | Unset):
        knowledge (list[KnowledgeDocument] | Unset):
    """

    name: str
    hash_: str
    instructions: str | Unset = UNSET
    skills: list[SkillRequest] | Unset = UNSET
    knowledge: list[KnowledgeDocument] | Unset = UNSET
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

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.knowledge_document import KnowledgeDocument
        from ..models.skill_request import SkillRequest

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

        sync_agent_request = cls(
            name=name,
            hash_=hash_,
            instructions=instructions,
            skills=skills,
            knowledge=knowledge,
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
