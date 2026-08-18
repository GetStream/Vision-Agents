from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SessionSkill")


@_attrs_define
class SessionSkill:
    """A kind of work worth handing to the slower model. There is nothing behind a skill but a better model: what it
    declares is the instructions that model answers under.

        Attributes:
            name (str):
            description (str): The one line the fast model sees.
            instructions (str): The full prompt, which only the subagent sees.
            deadline_ms (int | Unset): How long the work may run before it is abandoned. Zero is the default.
    """

    name: str
    description: str
    instructions: str
    deadline_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        description = self.description

        instructions = self.instructions

        deadline_ms = self.deadline_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "description": description,
                "instructions": instructions,
            }
        )
        if deadline_ms is not UNSET:
            field_dict["deadline_ms"] = deadline_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        description = d.pop("description")

        instructions = d.pop("instructions")

        deadline_ms = d.pop("deadline_ms", UNSET)

        session_skill = cls(
            name=name,
            description=description,
            instructions=instructions,
            deadline_ms=deadline_ms,
        )

        session_skill.additional_properties = d
        return session_skill

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
