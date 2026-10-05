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
        cases (int): How many conversations this run is having.
        failed (int):
        id (str):
        passed (int):
        simulation_id (str):
        started_at (datetime.datetime):
        state (SimulationRunState): A run passed only if every one of its conversations did. A conversation that never
            got as far as a ruling leaves the run errored rather than failed.
        assertion (str | Unset): What was asked of this run, copied when it started. Editing a simulation does not
            rewrite what an old run tested.
        config_id (str | Unset):
        conversations (list[SimulationCase] | Unset): The conversations this run had. Present when one run is asked for,
            and left out of a list so that reading the log does not mean reading every transcript.
        error (str | Unset):
        finished_at (datetime.datetime | Unset):
        judge_target (str | Unset):
        mode (SimulationRunMode | Unset):
        scenario (str | Unset):
    """

    cases: int
    failed: int
    id: str
    passed: int
    simulation_id: str
    started_at: datetime.datetime
    state: SimulationRunState
    assertion: str | Unset = UNSET
    config_id: str | Unset = UNSET
    conversations: list[SimulationCase] | Unset = UNSET
    error: str | Unset = UNSET
    finished_at: datetime.datetime | Unset = UNSET
    judge_target: str | Unset = UNSET
    mode: SimulationRunMode | Unset = UNSET
    scenario: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cases = self.cases

        failed = self.failed

        id = self.id

        passed = self.passed

        simulation_id = self.simulation_id

        started_at = self.started_at.isoformat()

        state = self.state.value

        assertion = self.assertion

        config_id = self.config_id

        conversations: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.conversations, Unset):
            conversations = []
            for conversations_item_data in self.conversations:
                conversations_item = conversations_item_data.to_dict()
                conversations.append(conversations_item)

        error = self.error

        finished_at: str | Unset = UNSET
        if not isinstance(self.finished_at, Unset):
            finished_at = self.finished_at.isoformat()

        judge_target = self.judge_target

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        scenario = self.scenario

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "cases": cases,
                "failed": failed,
                "id": id,
                "passed": passed,
                "simulation_id": simulation_id,
                "started_at": started_at,
                "state": state,
            }
        )
        if assertion is not UNSET:
            field_dict["assertion"] = assertion
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if conversations is not UNSET:
            field_dict["conversations"] = conversations
        if error is not UNSET:
            field_dict["error"] = error
        if finished_at is not UNSET:
            field_dict["finished_at"] = finished_at
        if judge_target is not UNSET:
            field_dict["judge_target"] = judge_target
        if mode is not UNSET:
            field_dict["mode"] = mode
        if scenario is not UNSET:
            field_dict["scenario"] = scenario

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.simulation_case import SimulationCase

        d = dict(src_dict)
        cases = d.pop("cases")

        failed = d.pop("failed")

        id = d.pop("id")

        passed = d.pop("passed")

        simulation_id = d.pop("simulation_id")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        state = SimulationRunState(d.pop("state"))

        assertion = d.pop("assertion", UNSET)

        config_id = d.pop("config_id", UNSET)

        _conversations = d.pop("conversations", UNSET)
        conversations: list[SimulationCase] | Unset = UNSET
        if _conversations is not UNSET:
            conversations = []
            for conversations_item_data in _conversations:
                conversations_item = SimulationCase.from_dict(conversations_item_data)

                conversations.append(conversations_item)

        error = d.pop("error", UNSET)

        _finished_at = d.pop("finished_at", UNSET)
        finished_at: datetime.datetime | Unset
        if isinstance(_finished_at, Unset):
            finished_at = UNSET
        else:
            finished_at = datetime.datetime.fromisoformat(_finished_at)

        judge_target = d.pop("judge_target", UNSET)

        _mode = d.pop("mode", UNSET)
        mode: SimulationRunMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = SimulationRunMode(_mode)

        scenario = d.pop("scenario", UNSET)

        simulation_run = cls(
            cases=cases,
            failed=failed,
            id=id,
            passed=passed,
            simulation_id=simulation_id,
            started_at=started_at,
            state=state,
            assertion=assertion,
            config_id=config_id,
            conversations=conversations,
            error=error,
            finished_at=finished_at,
            judge_target=judge_target,
            mode=mode,
            scenario=scenario,
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
