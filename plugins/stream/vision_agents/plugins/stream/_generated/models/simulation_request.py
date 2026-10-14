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
        assertion (str): What has to be true at the end for the run to have passed.
        config_id (str): The agent being tested.
        name (str):
        scenario (str): What to ask, in your own words and over as many turns as it takes. This is a brief for the
            caller rather than a script, so it may describe things that depend on what the agent says back.
        caller_stt (str | Unset): How the caller hears the agent. Audio simulations only.
        caller_target (str | Unset): The model that plays the caller. Empty takes llm-scenario-runner, the deployment's
            fast-tier default.
        caller_tts (str | Unset): How the caller speaks. Audio simulations only.
        caller_voice (str | Unset): The voice the caller speaks in. Audio simulations only.
        judge_target (str | Unset): The model that rules on the conversations, named the way any other routing target
            is. Empty takes llm-judge, the deployment's quality-tier default, since nobody is waiting for it.
        max_turns (int | Unset): How many times the caller may speak, up to two hundred. It is what stops a caller that
            never decides it is finished. Twelve when left out.
        mode (SimulationRequestMode | Unset): Text hands the agent the words, which tests everything between hearing and
            answering. Audio generates speech and runs the whole pipeline, so what is judged is what a caller would actually
            have heard. Text when left out.
        tags (SimulationRequestTags | Unset):
        variations (int | Unset): How many ways of asking the same thing one run tries, up to ten. The scenario as
            written is always the first of them, and one is what left out means.
    """

    assertion: str
    config_id: str
    name: str
    scenario: str
    caller_stt: str | Unset = UNSET
    caller_target: str | Unset = UNSET
    caller_tts: str | Unset = UNSET
    caller_voice: str | Unset = UNSET
    judge_target: str | Unset = UNSET
    max_turns: int | Unset = UNSET
    mode: SimulationRequestMode | Unset = UNSET
    tags: SimulationRequestTags | Unset = UNSET
    variations: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        assertion = self.assertion

        config_id = self.config_id

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
                "config_id": config_id,
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
        from ..models.simulation_request_tags import SimulationRequestTags

        d = dict(src_dict)
        assertion = d.pop("assertion")

        config_id = d.pop("config_id")

        name = d.pop("name")

        scenario = d.pop("scenario")

        caller_stt = d.pop("caller_stt", UNSET)

        caller_target = d.pop("caller_target", UNSET)

        caller_tts = d.pop("caller_tts", UNSET)

        caller_voice = d.pop("caller_voice", UNSET)

        judge_target = d.pop("judge_target", UNSET)

        max_turns = d.pop("max_turns", UNSET)

        _mode = d.pop("mode", UNSET)
        mode: SimulationRequestMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = SimulationRequestMode(_mode)

        _tags = d.pop("tags", UNSET)
        tags: SimulationRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = SimulationRequestTags.from_dict(_tags)

        variations = d.pop("variations", UNSET)

        simulation_request = cls(
            assertion=assertion,
            config_id=config_id,
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
