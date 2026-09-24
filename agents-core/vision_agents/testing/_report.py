"""Report model for ``vision-agents agent simulate``.

One ``SimulationRun`` per scenario and one ``SimulationCase`` per
conversation, built from the trials a ``Simulation`` produces. The shape
mirrors the hosted simulation API so local and hosted reports read the same.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from ._events import FunctionCallEvent, FunctionCallOutputEvent
from ._scenario import Scenario
from ._simulation import Trial, Turn

State = Literal["passed", "failed", "errored"]
Ended = Literal["complete", "turns", "failed"]


@dataclass
class ToolCall:
    """A tool the agent called while producing one reply."""

    name: str
    arguments: dict[str, Any]
    output: Any = None
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "output": self.output,
            "is_error": self.is_error,
        }


@dataclass
class TranscriptLine:
    """One message in a conversation.

    Attributes:
        caller: True when the simulated user said it rather than the agent.
        text: What was said; in audio mode, what the caller heard.
        latency_ms: Agent lines only; time from the user's message to the reply.
        tool_calls: Agent lines only; tools called while producing the reply.
    """

    caller: bool
    text: str
    latency_ms: float | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller": self.caller,
            "text": self.text,
            "latency_ms": self.latency_ms,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


@dataclass
class CriterionResult:
    """The judge's ruling on one success criterion."""

    criterion: str
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass
class SimulationCase:
    """One conversation: a single variation of a scenario, run once.

    Attributes:
        variation: Which wording of the brief this used; 0 is the scenario as written.
        attempt: Which repeat of that variation this was, starting at 0.
        scenario: The brief the simulated user followed.
        state: ``passed`` or ``failed`` by the judge, ``errored`` when the
            simulated user, a provider or the judge failed.
        turns: How many times the user spoke.
        ended: Why the conversation stopped: the user finished, ``max_turns``
            was hit (also reported when the user finished on exactly the
            last allowed turn), or it failed.
        criteria: The judge's ruling per criterion; empty when the case errored.
        error: What went wrong, for a failed or errored case.
    """

    id: str
    variation: int
    attempt: int
    scenario: str
    state: State
    transcript: list[TranscriptLine]
    turns: int
    ended: Ended
    started_at: datetime
    finished_at: datetime
    criteria: list[CriterionResult] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_trial(
        cls,
        trial: Trial,
        *,
        max_turns: int,
        started_at: datetime,
        finished_at: datetime,
    ) -> "SimulationCase":
        """Describe a judged trial as a case."""
        state: State
        if not trial.valid:
            state = "errored"
        elif trial.passed:
            state = "passed"
        else:
            state = "failed"
        ended: Ended
        if trial.error is not None:
            ended = "failed"
        elif trial.turn_count >= max_turns:
            ended = "turns"
        else:
            ended = "complete"
        return cls(
            id=uuid4().hex,
            variation=trial.variation,
            attempt=trial.repeat,
            scenario=trial.scenario.brief,
            state=state,
            transcript=[line for turn in trial.turns for line in _lines(turn)],
            turns=trial.turn_count,
            ended=ended,
            started_at=started_at,
            finished_at=finished_at,
            criteria=[
                CriterionResult(criterion=name, passed=v.success, reason=v.reason)
                for name, v in trial.verdicts.items()
            ],
            error=trial.error,
        )

    @property
    def passed(self) -> bool | None:
        """The judge's ruling; ``None`` when it never got as far as ruling."""
        if self.state == "errored":
            return None
        return self.state == "passed"

    @property
    def failed_criteria(self) -> list[str]:
        return [v.criterion for v in self.criteria if not v.passed]

    @property
    def verdict(self) -> str | None:
        """What decided the case, in the judge's words."""
        if self.state == "errored":
            return None
        failed = [f"{v.criterion}: {v.reason}" for v in self.criteria if not v.passed]
        return "; ".join(failed) if failed else "All criteria met."

    @property
    def agent_latencies_ms(self) -> list[float]:
        return [
            line.latency_ms
            for line in self.transcript
            if not line.caller and line.latency_ms is not None
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "variation": self.variation,
            "attempt": self.attempt,
            "scenario": self.scenario,
            "state": self.state,
            "transcript": [line.to_dict() for line in self.transcript],
            "turns": self.turns,
            "passed": self.passed,
            "verdict": self.verdict,
            "criteria": [v.to_dict() for v in self.criteria],
            "ended": self.ended,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
        }


@dataclass
class SimulationRun:
    """Every conversation one scenario had.

    A run passed only if every one of its conversations did. A conversation
    that never got as far as a ruling leaves the run errored rather than failed.
    """

    id: str
    name: str
    scenario: str
    criteria: list[str]
    mode: str
    variations: int
    repeat: int
    max_turns: int
    judge_target: str
    conversations: list[SimulationCase]
    started_at: datetime
    finished_at: datetime
    error: str | None = None
    pass_at_k: float | None = None

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        *,
        conversations: list[SimulationCase],
        max_turns: int,
        judge_target: str,
        started_at: datetime,
        finished_at: datetime,
        error: str | None = None,
        pass_at_k: float | None = None,
    ) -> "SimulationRun":
        return cls(
            id=uuid4().hex,
            name=scenario.name,
            scenario=scenario.brief,
            criteria=list(scenario.success),
            mode=scenario.mode,
            variations=scenario.variations,
            repeat=scenario.repeat,
            max_turns=max_turns,
            judge_target=judge_target,
            conversations=conversations,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
            pass_at_k=pass_at_k,
        )

    @property
    def state(self) -> State:
        if self.error is not None or self.errored:
            return "errored"
        if self.failed:
            return "failed"
        return "passed"

    @property
    def cases(self) -> int:
        return len(self.conversations)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.conversations if c.state == "passed")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.conversations if c.state == "failed")

    @property
    def errored(self) -> int:
        return sum(1 for c in self.conversations if c.state == "errored")

    @property
    def failed_criteria(self) -> list[str]:
        """Criteria that failed in any conversation, in scenario order."""
        failed = {c for case in self.conversations for c in case.failed_criteria}
        return [c for c in self.criteria if c in failed]

    @property
    def mean_turns(self) -> float | None:
        turns = [c.turns for c in self.conversations]
        return statistics.mean(turns) if turns else None

    @property
    def p50_latency_ms(self) -> float | None:
        """Median time the agent took to reply, across all conversations."""
        latencies = [ms for c in self.conversations for ms in c.agent_latencies_ms]
        return statistics.median(latencies) if latencies else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "scenario": self.scenario,
            "criteria": self.criteria,
            "mode": self.mode,
            "variations": self.variations,
            "repeat": self.repeat,
            "max_turns": self.max_turns,
            "judge_target": self.judge_target,
            "state": self.state,
            "cases": self.cases,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "pass_at_k": self.pass_at_k,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "conversations": [c.to_dict() for c in self.conversations],
        }


@dataclass
class SimulationReport:
    """Everything one ``simulate`` invocation ran."""

    runs: list[SimulationRun]
    judge_target: str
    started_at: datetime
    finished_at: datetime

    @property
    def state(self) -> State:
        states = {run.state for run in self.runs}
        if "errored" in states:
            return "errored"
        if "failed" in states:
            return "failed"
        return "passed"

    @property
    def exit_code(self) -> int:
        """0 when everything passed, 1 when a scenario failed, 2 when something errored."""
        return {"passed": 0, "failed": 1, "errored": 2}[self.state]

    @property
    def cases(self) -> int:
        return sum(run.cases for run in self.runs)

    @property
    def passed(self) -> int:
        return sum(run.passed for run in self.runs)

    @property
    def failed(self) -> int:
        return sum(run.failed for run in self.runs)

    @property
    def errored(self) -> int:
        return sum(run.errored for run in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "judge_target": self.judge_target,
            "cases": self.cases,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "runs": [run.to_dict() for run in self.runs],
        }


def _lines(turn: Turn) -> list[TranscriptLine]:
    """The user's line and the agent's reply, with tool calls paired to their results."""
    calls: list[ToolCall] = []
    call_ids: list[str | None] = []
    answered: list[bool] = []
    for event in turn.response.events:
        if isinstance(event, FunctionCallEvent):
            calls.append(ToolCall(name=event.name, arguments=dict(event.arguments)))
            call_ids.append(event.tool_call_id)
            answered.append(False)
        elif isinstance(event, FunctionCallOutputEvent):
            for i, call in enumerate(calls):
                if answered[i]:
                    continue
                same_id = (
                    event.tool_call_id is not None and call_ids[i] == event.tool_call_id
                )
                same_name = event.tool_call_id is None and call.name == event.name
                if same_id or same_name:
                    call.output = event.output
                    call.is_error = event.is_error
                    answered[i] = True
                    break
    return [
        TranscriptLine(caller=True, text=turn.user_message),
        TranscriptLine(
            caller=False,
            text=turn.agent_reply or "",
            latency_ms=turn.latency_ms,
            tool_calls=calls,
        ),
    ]
