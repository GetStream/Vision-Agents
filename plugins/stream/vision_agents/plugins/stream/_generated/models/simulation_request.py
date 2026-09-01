from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.simulation_request_mode import SimulationRequestMode
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.simulation_request_tags import SimulationRequestTags


T = TypeVar("T", bound="SimulationRequest")


@_attrs_define
class SimulationRequest:
    """
    Attributes:
        name (str):
        config_id (str): The agent being tested.
        scenario (str): What to ask, in your own words and over as many turns as it takes. This is a brief for the
            caller rather than a script, so it may describe things that depend on what the agent says back.
        assertion (str): What has to be true at the end for the run to have passed.
        mode (SimulationRequestMode | Unset): Text hands the agent the words, which tests everything between hearing and
            answering. Audio generates speech and runs the whole pipeline, so what is judged is what a caller would actually
            have heard.
             Default: SimulationRequestMode.TEXT.
        variations (int | Unset): How many ways of asking the same thing one run tries, up to ten. The scenario as
            written is always the first of them.
             Default: 1.
        judge_target (str | Unset): The model that rules on the conversations, named the way any other routing target
            is. Empty takes a quality tier, since nobody is waiting for it.
        caller_target (str | Unset): The model that plays the caller. Empty takes a fast tier.
        caller_tts (str | Unset): How the caller speaks. Audio simulations only.
        caller_stt (str | Unset): How the caller hears the agent. Audio simulations only.
        caller_voice (str | Unset): The voice the caller speaks in. Audio simulations only.
        max_turns (int | Unset): How many times the caller may speak, up to thirty. It is what stops a caller that never
            decides it is finished.
             Default: 12.
        tags (SimulationRequestTags | Unset):
    """

    name: str
    config_id: str
    scenario: str
    assertion: str
    mode: SimulationRequestMode | Unset = SimulationRequestMode.TEXT
    variations: int | Unset = 1
    judge_target: str | Unset = UNSET
    caller_target: str | Unset = UNSET
    caller_tts: str | Unset = UNSET
    caller_stt: str | Unset = UNSET
    caller_voice: str | Unset = UNSET
    max_turns: int | Unset = 12
    tags: SimulationRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        config_id = self.config_id

        scenario = self.scenario

        assertion = self.assertion

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        variations = self.variations

        judge_target = self.judge_target

        caller_target = self.caller_target

        caller_tts = self.caller_tts

        caller_stt = self.caller_stt

        caller_voice = self.caller_voice

        max_turns = self.max_turns

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "config_id": config_id,
                "scenario": scenario,
                "assertion": assertion,
            }
        )
        if mode is not UNSET:
            field_dict["mode"] = mode
        if variations is not UNSET:
            field_dict["variations"] = variations
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
        if max_turns is not UNSET:
            field_dict["max_turns"] = max_turns
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.simulation_request_tags import (
            SimulationRequestTags,
        )

        d = dict(src_dict)
        name = d.pop("name")

        config_id = d.pop("config_id")

        scenario = d.pop("scenario")

        assertion = d.pop("assertion")

        _mode = d.pop("mode", UNSET)
        mode: SimulationRequestMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = SimulationRequestMode(_mode)

        variations = d.pop("variations", UNSET)

        judge_target = d.pop("judge_target", UNSET)

        caller_target = d.pop("caller_target", UNSET)

        caller_tts = d.pop("caller_tts", UNSET)

        caller_stt = d.pop("caller_stt", UNSET)

        caller_voice = d.pop("caller_voice", UNSET)

        max_turns = d.pop("max_turns", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: SimulationRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = SimulationRequestTags.from_dict(_tags)

        simulation_request = cls(
            name=name,
            config_id=config_id,
            scenario=scenario,
            assertion=assertion,
            mode=mode,
            variations=variations,
            judge_target=judge_target,
            caller_target=caller_target,
            caller_tts=caller_tts,
            caller_stt=caller_stt,
            caller_voice=caller_voice,
            max_turns=max_turns,
            tags=tags,
        )

        simulation_request.additional_properties = d
        return simulation_request

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
