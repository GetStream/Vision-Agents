from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.simulation_case_ended import SimulationCaseEnded
from ..models.simulation_case_state import SimulationCaseState
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.simulation_line import SimulationLine


T = TypeVar("T", bound="SimulationCase")


@_attrs_define
class SimulationCase:
    """
    Attributes:
        id (str):
        variation (int): Which way of asking this was, and the order they are listed in.
        scenario (str): The wording this conversation used.
        state (SimulationCaseState):
        turns (int): How many times the caller spoke.
        started_at (datetime.datetime):
        call_id (str | Unset): The session that held it, which is what the call and transcript paths take. It is written
            as soon as it exists, so a conversation still going can be watched.
        transcript (list[SimulationLine] | Unset):
        passed (bool | Unset): The judge's ruling. Absent when it never got as far as ruling, which is not the same as
            having ruled against.
        verdict (str | Unset): What in the conversation decided it.
        score (int | Unset): How sure the judge was, from 1 to 5.
        ended (SimulationCaseEnded | Unset): Why the conversation stopped.
        error (str | Unset):
        finished_at (datetime.datetime | Unset):
    """

    id: str
    variation: int
    scenario: str
    state: SimulationCaseState
    turns: int
    started_at: datetime.datetime
    call_id: str | Unset = UNSET
    transcript: list[SimulationLine] | Unset = UNSET
    passed: bool | Unset = UNSET
    verdict: str | Unset = UNSET
    score: int | Unset = UNSET
    ended: SimulationCaseEnded | Unset = UNSET
    error: str | Unset = UNSET
    finished_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        variation = self.variation

        scenario = self.scenario

        state = self.state.value

        turns = self.turns

        started_at = self.started_at.isoformat()

        call_id = self.call_id

        transcript: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.transcript, Unset):
            transcript = []
            for transcript_item_data in self.transcript:
                transcript_item = transcript_item_data.to_dict()
                transcript.append(transcript_item)

        passed = self.passed

        verdict = self.verdict

        score = self.score

        ended: str | Unset = UNSET
        if not isinstance(self.ended, Unset):
            ended = self.ended.value

        error = self.error

        finished_at: str | Unset = UNSET
        if not isinstance(self.finished_at, Unset):
            finished_at = self.finished_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "variation": variation,
                "scenario": scenario,
                "state": state,
                "turns": turns,
                "started_at": started_at,
            }
        )
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if transcript is not UNSET:
            field_dict["transcript"] = transcript
        if passed is not UNSET:
            field_dict["passed"] = passed
        if verdict is not UNSET:
            field_dict["verdict"] = verdict
        if score is not UNSET:
            field_dict["score"] = score
        if ended is not UNSET:
            field_dict["ended"] = ended
        if error is not UNSET:
            field_dict["error"] = error
        if finished_at is not UNSET:
            field_dict["finished_at"] = finished_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.simulation_line import SimulationLine

        d = dict(src_dict)
        id = d.pop("id")

        variation = d.pop("variation")

        scenario = d.pop("scenario")

        state = SimulationCaseState(d.pop("state"))

        turns = d.pop("turns")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        call_id = d.pop("call_id", UNSET)

        _transcript = d.pop("transcript", UNSET)
        transcript: list[SimulationLine] | Unset = UNSET
        if _transcript is not UNSET:
            transcript = []
            for transcript_item_data in _transcript:
                transcript_item = SimulationLine.from_dict(transcript_item_data)

                transcript.append(transcript_item)

        passed = d.pop("passed", UNSET)

        verdict = d.pop("verdict", UNSET)

        score = d.pop("score", UNSET)

        _ended = d.pop("ended", UNSET)
        ended: SimulationCaseEnded | Unset
        if isinstance(_ended, Unset):
            ended = UNSET
        else:
            ended = SimulationCaseEnded(_ended)

        error = d.pop("error", UNSET)

        _finished_at = d.pop("finished_at", UNSET)
        finished_at: datetime.datetime | Unset
        if isinstance(_finished_at, Unset):
            finished_at = UNSET
        else:
            finished_at = datetime.datetime.fromisoformat(_finished_at)

        simulation_case = cls(
            id=id,
            variation=variation,
            scenario=scenario,
            state=state,
            turns=turns,
            started_at=started_at,
            call_id=call_id,
            transcript=transcript,
            passed=passed,
            verdict=verdict,
            score=score,
            ended=ended,
            error=error,
            finished_at=finished_at,
        )

        simulation_case.additional_properties = d
        return simulation_case

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
