from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="Skill")


@_attrs_define
class Skill:
    """
    Attributes:
        id (str):
        config_id (str):
        name (str):
        description (str):
        instructions (str):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        subagent (str | Unset): Named worker binding; omitted uses default.
        capture_video (bool | Unset): Capture task-scoped visual evidence before reasoning.
        deadline_ms (int | Unset):
    """

    id: str
    config_id: str
    name: str
    description: str
    instructions: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    subagent: str | Unset = UNSET
    capture_video: bool | Unset = UNSET
    deadline_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        config_id = self.config_id

        name = self.name

        description = self.description

        instructions = self.instructions

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        subagent = self.subagent

        capture_video = self.capture_video

        deadline_ms = self.deadline_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "config_id": config_id,
                "name": name,
                "description": description,
                "instructions": instructions,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if capture_video is not UNSET:
            field_dict["capture_video"] = capture_video
        if deadline_ms is not UNSET:
            field_dict["deadline_ms"] = deadline_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        id = d.pop("id")

        config_id = d.pop("config_id")

        name = d.pop("name")

        description = d.pop("description")

        instructions = d.pop("instructions")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        subagent = d.pop("subagent", UNSET)

        capture_video = d.pop("capture_video", UNSET)

        deadline_ms = d.pop("deadline_ms", UNSET)

        skill = cls(
            id=id,
            config_id=config_id,
            name=name,
            description=description,
            instructions=instructions,
            created_at=created_at,
            updated_at=updated_at,
            subagent=subagent,
            capture_video=capture_video,
            deadline_ms=deadline_ms,
        )

        skill.additional_properties = d
        return skill

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
