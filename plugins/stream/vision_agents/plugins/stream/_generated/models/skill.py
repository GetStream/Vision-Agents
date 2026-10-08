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
        config_id (str):
        created_at (datetime.datetime):
        description (str):
        id (str):
        instructions (str):
        name (str):
        updated_at (datetime.datetime):
        capture_video (bool | Unset): Capture task-scoped visual evidence before reasoning.
        deadline_ms (int | Unset):
    """

    config_id: str
    created_at: datetime.datetime
    description: str
    id: str
    instructions: str
    name: str
    updated_at: datetime.datetime
    capture_video: bool | Unset = UNSET
    deadline_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        config_id = self.config_id

        created_at = self.created_at.isoformat()

        description = self.description

        id = self.id

        instructions = self.instructions

        name = self.name

        updated_at = self.updated_at.isoformat()

        capture_video = self.capture_video

        deadline_ms = self.deadline_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "config_id": config_id,
                "created_at": created_at,
                "description": description,
                "id": id,
                "instructions": instructions,
                "name": name,
                "updated_at": updated_at,
            }
        )
        if capture_video is not UNSET:
            field_dict["capture_video"] = capture_video
        if deadline_ms is not UNSET:
            field_dict["deadline_ms"] = deadline_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        config_id = d.pop("config_id")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        description = d.pop("description")

        id = d.pop("id")

        instructions = d.pop("instructions")

        name = d.pop("name")

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        capture_video = d.pop("capture_video", UNSET)

        deadline_ms = d.pop("deadline_ms", UNSET)

        skill = cls(
            config_id=config_id,
            created_at=created_at,
            description=description,
            id=id,
            instructions=instructions,
            name=name,
            updated_at=updated_at,
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
