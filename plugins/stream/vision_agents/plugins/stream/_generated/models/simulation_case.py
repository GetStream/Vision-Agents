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
        scenario (str): The wording this conversation used.
        started_at (datetime.datetime):
        state (SimulationCaseState):
        turns (int): How many times the caller spoke.
        variation (int): Which way of asking this was, and the order they are listed in.
        call_id (str | Unset): The session that held it, which is what the call and transcript paths take. It is written
            as soon as it exists, so a conversation still going can be watched.
        ended (SimulationCaseEnded | Unset): Why the conversation stopped.
        error (str | Unset):
        finished_at (datetime.datetime | Unset):
        passed (bool | Unset): The judge's ruling. Absent when it never got as far as ruling, which is not the same as
            having ruled against.
        score (int | Unset): How sure the judge was, from 1 to 5.
        transcript (list[SimulationLine] | Unset):
        verdict (str | Unset): What in the conversation decided it.
    """

    id: str
    scenario: str
    started_at: datetime.datetime
    state: SimulationCaseState
    turns: int
    variation: int
    call_id: str | Unset = UNSET
    ended: SimulationCaseEnded | Unset = UNSET
    error: str | Unset = UNSET
    finished_at: datetime.datetime | Unset = UNSET
    passed: bool | Unset = UNSET
    score: int | Unset = UNSET
    transcript: list[SimulationLine] | Unset = UNSET
    verdict: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        scenario = self.scenario

        started_at = self.started_at.isoformat()

        state = self.state.value

        turns = self.turns

        variation = self.variation

        call_id = self.call_id

        ended: str | Unset = UNSET
        if not isinstance(self.ended, Unset):
            ended = self.ended.value

        error = self.error

        finished_at: str | Unset = UNSET
        if not isinstance(self.finished_at, Unset):
            finished_at = self.finished_at.isoformat()

        passed = self.passed

        score = self.score

        transcript: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.transcript, Unset):
            transcript = []
            for transcript_item_data in self.transcript:
                transcript_item = transcript_item_data.to_dict()
                transcript.append(transcript_item)

        verdict = self.verdict

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "scenario": scenario,
                "started_at": started_at,
                "state": state,
                "turns": turns,
                "variation": variation,
            }
        )
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if ended is not UNSET:
            field_dict["ended"] = ended
        if error is not UNSET:
            field_dict["error"] = error
        if finished_at is not UNSET:
            field_dict["finished_at"] = finished_at
        if passed is not UNSET:
            field_dict["passed"] = passed
        if score is not UNSET:
            field_dict["score"] = score
        if transcript is not UNSET:
            field_dict["transcript"] = transcript
        if verdict is not UNSET:
            field_dict["verdict"] = verdict

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.simulation_line import SimulationLine

        d = dict(src_dict)
        id = d.pop("id")

        scenario = d.pop("scenario")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        state = SimulationCaseState(d.pop("state"))

        turns = d.pop("turns")

        variation = d.pop("variation")

        call_id = d.pop("call_id", UNSET)

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

        passed = d.pop("passed", UNSET)

        score = d.pop("score", UNSET)

        _transcript = d.pop("transcript", UNSET)
        transcript: list[SimulationLine] | Unset = UNSET
        if _transcript is not UNSET:
            transcript = []
            for transcript_item_data in _transcript:
                transcript_item = SimulationLine.from_dict(transcript_item_data)

                transcript.append(transcript_item)

        verdict = d.pop("verdict", UNSET)

        simulation_case = cls(
            id=id,
            scenario=scenario,
            started_at=started_at,
            state=state,
            turns=turns,
            variation=variation,
            call_id=call_id,
            ended=ended,
            error=error,
            finished_at=finished_at,
            passed=passed,
            score=score,
            transcript=transcript,
            verdict=verdict,
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
