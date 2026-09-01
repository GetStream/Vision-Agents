from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.simulation_mode import SimulationMode
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.simulation_tags import SimulationTags


T = TypeVar("T", bound="Simulation")


@_attrs_define
class Simulation:
    """
    Attributes:
        id (str):
        name (str):
        mode (SimulationMode):
        config_id (str):
        scenario (str):
        assertion (str):
        variations (int):
        max_turns (int):
        created_at (datetime.datetime):
        judge_target (str | Unset):
        caller_target (str | Unset):
        caller_tts (str | Unset):
        caller_stt (str | Unset):
        caller_voice (str | Unset):
        tags (SimulationTags | Unset):
        updated_at (datetime.datetime | Unset):
    """

    id: str
    name: str
    mode: SimulationMode
    config_id: str
    scenario: str
    assertion: str
    variations: int
    max_turns: int
    created_at: datetime.datetime
    judge_target: str | Unset = UNSET
    caller_target: str | Unset = UNSET
    caller_tts: str | Unset = UNSET
    caller_stt: str | Unset = UNSET
    caller_voice: str | Unset = UNSET
    tags: SimulationTags | Unset = UNSET
    updated_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        mode = self.mode.value

        config_id = self.config_id

        scenario = self.scenario

        assertion = self.assertion

        variations = self.variations

        max_turns = self.max_turns

        created_at = self.created_at.isoformat()

        judge_target = self.judge_target

        caller_target = self.caller_target

        caller_tts = self.caller_tts

        caller_stt = self.caller_stt

        caller_voice = self.caller_voice

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        updated_at: str | Unset = UNSET
        if not isinstance(self.updated_at, Unset):
            updated_at = self.updated_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "mode": mode,
                "config_id": config_id,
                "scenario": scenario,
                "assertion": assertion,
                "variations": variations,
                "max_turns": max_turns,
                "created_at": created_at,
            }
        )
        if judge_target is not UNSET:
            field_dict["judge_target"] = judge_target
        if caller_target is not UNSET:
            field_dict["caller_target"] = caller_target
        if caller_tts is not UNSET:
            field_dict["caller_tts"] = caller_tts
        if caller_stt is not UNSET:
            field_dict["caller_stt"] = caller_stt
        if caller_voice is not UNSET:
            field_dict["caller_voice"] = caller_voice
        if tags is not UNSET:
            field_dict["tags"] = tags
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.simulation_tags import SimulationTags

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        mode = SimulationMode(d.pop("mode"))

        config_id = d.pop("config_id")

        scenario = d.pop("scenario")

        assertion = d.pop("assertion")

        variations = d.pop("variations")

        max_turns = d.pop("max_turns")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        judge_target = d.pop("judge_target", UNSET)

        caller_target = d.pop("caller_target", UNSET)

        caller_tts = d.pop("caller_tts", UNSET)

        caller_stt = d.pop("caller_stt", UNSET)

        caller_voice = d.pop("caller_voice", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: SimulationTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = SimulationTags.from_dict(_tags)

        _updated_at = d.pop("updated_at", UNSET)
        updated_at: datetime.datetime | Unset
        if isinstance(_updated_at, Unset):
            updated_at = UNSET
        else:
            updated_at = datetime.datetime.fromisoformat(_updated_at)

        simulation = cls(
            id=id,
            name=name,
            mode=mode,
            config_id=config_id,
            scenario=scenario,
            assertion=assertion,
            variations=variations,
            max_turns=max_turns,
            created_at=created_at,
            judge_target=judge_target,
            caller_target=caller_target,
            caller_tts=caller_tts,
            caller_stt=caller_stt,
            caller_voice=caller_voice,
            tags=tags,
            updated_at=updated_at,
        )

        simulation.additional_properties = d
        return simulation

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
