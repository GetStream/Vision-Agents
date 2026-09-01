from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.simulation_run_mode import SimulationRunMode
from ..models.simulation_run_state import SimulationRunState
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.simulation_case import SimulationCase


T = TypeVar("T", bound="SimulationRun")


@_attrs_define
class SimulationRun:
    """
    Attributes:
        id (str):
        simulation_id (str):
        state (SimulationRunState): A run passed only if every one of its conversations did. A conversation that never
            got as far as a ruling leaves the run errored rather than failed.
        cases (int): How many conversations this run is having.
        passed (int):
        failed (int):
        started_at (datetime.datetime):
        mode (SimulationRunMode | Unset):
        config_id (str | Unset):
        scenario (str | Unset):
        assertion (str | Unset): What was asked of this run, copied when it started. Editing a simulation does not
            rewrite what an old run tested.
        judge_target (str | Unset):
        error (str | Unset):
        finished_at (datetime.datetime | Unset):
        conversations (list[SimulationCase] | Unset): The conversations this run had. Present when one run is asked for,
            and left out of a list so that reading the log does not mean reading every transcript.
    """

    id: str
    simulation_id: str
    state: SimulationRunState
    cases: int
    passed: int
    failed: int
    started_at: datetime.datetime
    mode: SimulationRunMode | Unset = UNSET
    config_id: str | Unset = UNSET
    scenario: str | Unset = UNSET
    assertion: str | Unset = UNSET
    judge_target: str | Unset = UNSET
    error: str | Unset = UNSET
    finished_at: datetime.datetime | Unset = UNSET
    conversations: list[SimulationCase] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        simulation_id = self.simulation_id

        state = self.state.value

        cases = self.cases

        passed = self.passed

        failed = self.failed

        started_at = self.started_at.isoformat()

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        config_id = self.config_id

        scenario = self.scenario

        assertion = self.assertion

        judge_target = self.judge_target

        error = self.error

        finished_at: str | Unset = UNSET
        if not isinstance(self.finished_at, Unset):
            finished_at = self.finished_at.isoformat()

        conversations: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.conversations, Unset):
            conversations = []
            for conversations_item_data in self.conversations:
                conversations_item = conversations_item_data.to_dict()
                conversations.append(conversations_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "simulation_id": simulation_id,
                "state": state,
                "cases": cases,
                "passed": passed,
                "failed": failed,
                "started_at": started_at,
            }
        )
        if mode is not UNSET:
            field_dict["mode"] = mode
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if scenario is not UNSET:
            field_dict["scenario"] = scenario
        if assertion is not UNSET:
            field_dict["assertion"] = assertion
        if judge_target is not UNSET:
            field_dict["judge_target"] = judge_target
        if error is not UNSET:
            field_dict["error"] = error
        if finished_at is not UNSET:
            field_dict["finished_at"] = finished_at
        if conversations is not UNSET:
            field_dict["conversations"] = conversations

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.simulation_case import SimulationCase

        d = dict(src_dict)
        id = d.pop("id")

        simulation_id = d.pop("simulation_id")

        state = SimulationRunState(d.pop("state"))

        cases = d.pop("cases")

        passed = d.pop("passed")

        failed = d.pop("failed")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        _mode = d.pop("mode", UNSET)
        mode: SimulationRunMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = SimulationRunMode(_mode)

        config_id = d.pop("config_id", UNSET)

        scenario = d.pop("scenario", UNSET)

        assertion = d.pop("assertion", UNSET)

        judge_target = d.pop("judge_target", UNSET)

        error = d.pop("error", UNSET)

        _finished_at = d.pop("finished_at", UNSET)
        finished_at: datetime.datetime | Unset
        if isinstance(_finished_at, Unset):
            finished_at = UNSET
        else:
            finished_at = datetime.datetime.fromisoformat(_finished_at)

        _conversations = d.pop("conversations", UNSET)
        conversations: list[SimulationCase] | Unset = UNSET
        if _conversations is not UNSET:
            conversations = []
            for conversations_item_data in _conversations:
                conversations_item = SimulationCase.from_dict(conversations_item_data)

                conversations.append(conversations_item)

        simulation_run = cls(
            id=id,
            simulation_id=simulation_id,
            state=state,
            cases=cases,
            passed=passed,
            failed=failed,
            started_at=started_at,
            mode=mode,
            config_id=config_id,
            scenario=scenario,
            assertion=assertion,
            judge_target=judge_target,
            error=error,
            finished_at=finished_at,
            conversations=conversations,
        )

        simulation_run.additional_properties = d
        return simulation_run

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
