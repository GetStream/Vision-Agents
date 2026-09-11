from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SkillRequest")


@_attrs_define
class SkillRequest:
    """
    Attributes:
        config_id (str): The agent config this skill belongs to. A skill is not shared: two agents that both need one
            have one each, so editing either leaves the other alone.
        name (str): How the config names it, which is unique among that config's own skills.
        description (str): The one line the fast model sees.
        instructions (str): The full prompt, which only the subagent sees.
        subagent (str | Unset): Named worker binding; omitted uses default.
        capture_video (bool | Unset): Capture task-scoped visual evidence before reasoning.
        deadline_ms (int | Unset): How long the work may run before it is abandoned. Zero is the default.
    """

    config_id: str
    name: str
    description: str
    instructions: str
    subagent: str | Unset = UNSET
    capture_video: bool | Unset = UNSET
    deadline_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        config_id = self.config_id

        name = self.name

        description = self.description

        instructions = self.instructions

        subagent = self.subagent

        capture_video = self.capture_video

        deadline_ms = self.deadline_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "config_id": config_id,
                "name": name,
                "description": description,
                "instructions": instructions,
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
        config_id = d.pop("config_id")

        name = d.pop("name")

        description = d.pop("description")

        instructions = d.pop("instructions")

        subagent = d.pop("subagent", UNSET)

        capture_video = d.pop("capture_video", UNSET)

        deadline_ms = d.pop("deadline_ms", UNSET)

        skill_request = cls(
            config_id=config_id,
            name=name,
            description=description,
            instructions=instructions,
            subagent=subagent,
            capture_video=capture_video,
            deadline_ms=deadline_ms,
        )

        skill_request.additional_properties = d
        return skill_request

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
