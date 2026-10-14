from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.simulation_declaration_mode import SimulationDeclarationMode
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.simulation_declaration_tags import SimulationDeclarationTags


T = TypeVar("T", bound="SimulationDeclaration")


@_attrs_define
class SimulationDeclaration:
    """A simulation an agent directory declares in simulations/*.yaml. It runs against the agent being synced.

    Attributes:
        assertion (str): What has to be true at the end for a run to have passed.
        name (str): Unique among the agent's simulations, and what a sync finds it again by.
        scenario (str): What the caller wants, in your own words and over as many turns as it takes.
        caller_stt (str | Unset):
        caller_target (str | Unset):
        caller_tts (str | Unset):
        caller_voice (str | Unset):
        judge_target (str | Unset):
        max_turns (int | Unset): How many times the caller may speak. Twelve when left out.
        mode (SimulationDeclarationMode | Unset): Text when left out.
        tags (SimulationDeclarationTags | Unset):
        variations (int | Unset): How many ways of asking the same thing one run tries.
    """

    assertion: str
    name: str
    scenario: str
    caller_stt: str | Unset = UNSET
    caller_target: str | Unset = UNSET
    caller_tts: str | Unset = UNSET
    caller_voice: str | Unset = UNSET
    judge_target: str | Unset = UNSET
    max_turns: int | Unset = UNSET
    mode: SimulationDeclarationMode | Unset = UNSET
    tags: SimulationDeclarationTags | Unset = UNSET
    variations: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        assertion = self.assertion

        name = self.name

        scenario = self.scenario

        caller_stt = self.caller_stt

        caller_target = self.caller_target

        caller_tts = self.caller_tts

        caller_voice = self.caller_voice

        judge_target = self.judge_target

        max_turns = self.max_turns

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        variations = self.variations

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "assertion": assertion,
                "name": name,
                "scenario": scenario,
            }
        )
        if caller_stt is not UNSET:
            field_dict["caller_stt"] = caller_stt
        if caller_target is not UNSET:
            field_dict["caller_target"] = caller_target
        if caller_tts is not UNSET:
            field_dict["caller_tts"] = caller_tts
        if caller_voice is not UNSET:
            field_dict["caller_voice"] = caller_voice
        if judge_target is not UNSET:
            field_dict["judge_target"] = judge_target
        if max_turns is not UNSET:
            field_dict["max_turns"] = max_turns
        if mode is not UNSET:
            field_dict["mode"] = mode
        if tags is not UNSET:
            field_dict["tags"] = tags
        if variations is not UNSET:
            field_dict["variations"] = variations

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.simulation_declaration_tags import SimulationDeclarationTags

        d = dict(src_dict)
        assertion = d.pop("assertion")

        name = d.pop("name")

        scenario = d.pop("scenario")

        caller_stt = d.pop("caller_stt", UNSET)

        caller_target = d.pop("caller_target", UNSET)

        caller_tts = d.pop("caller_tts", UNSET)

        caller_voice = d.pop("caller_voice", UNSET)

        judge_target = d.pop("judge_target", UNSET)

        max_turns = d.pop("max_turns", UNSET)

        _mode = d.pop("mode", UNSET)
        mode: SimulationDeclarationMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = SimulationDeclarationMode(_mode)

        _tags = d.pop("tags", UNSET)
        tags: SimulationDeclarationTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = SimulationDeclarationTags.from_dict(_tags)

        variations = d.pop("variations", UNSET)

        simulation_declaration = cls(
            assertion=assertion,
            name=name,
            scenario=scenario,
            caller_stt=caller_stt,
            caller_target=caller_target,
            caller_tts=caller_tts,
            caller_voice=caller_voice,
            judge_target=judge_target,
            max_turns=max_turns,
            mode=mode,
            tags=tags,
            variations=variations,
        )

        simulation_declaration.additional_properties = d
        return simulation_declaration

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
