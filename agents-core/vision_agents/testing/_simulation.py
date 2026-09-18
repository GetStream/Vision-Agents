"""Multi-turn simulation runner: simulated user vs. agent, judged per criterion."""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from math import comb
from typing import TypeVar

from vision_agents.core.agents.agents import Agent
from vision_agents.core.llm.llm import LLM

from ._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from ._judge import Judge, JudgeError, JudgeVerdict
from ._run_result import TestResponse
from ._scenario import Scenario
from ._session import TestSession
from ._simulated_user import SimulatedUser, SimulatedUserError
from ._variations import generate_variations

logger = logging.getLogger(__name__)

T = TypeVar("T")

Target = Agent | LLM
TargetFactory = Callable[[], Target]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate that at least one of ``k`` trials passes.

    Args:
        n: Number of valid trials run.
        c: Number of those trials that passed.
        k: Number of attempts the estimate is for. Must not exceed ``n``.
    """
    if k < 1 or n < k or c < 0 or c > n:
        raise ValueError(f"Invalid pass@k inputs: n={n}, c={c}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_pow_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate that all of ``k`` trials pass.

    Args:
        n: Number of valid trials run.
        c: Number of those trials that passed.
        k: Number of attempts the estimate is for. Must not exceed ``n``.
    """
    if k < 1 or n < k or c < 0 or c > n:
        raise ValueError(f"Invalid pass^k inputs: n={n}, c={c}, k={k}")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def render_transcript(events: list[RunEvent]) -> str:
    """Render conversation events as plain text for a judge or a report."""
    lines: list[str] = []
    for event in events:
        if isinstance(event, ChatMessageEvent):
            lines.append(f"[{event.role}] {event.content}")
        elif isinstance(event, FunctionCallEvent):
            lines.append(
                f"[agent called tool {event.name}] {json.dumps(event.arguments, default=str)}"
            )
        elif isinstance(event, FunctionCallOutputEvent):
            status = "failed" if event.is_error else "returned"
            lines.append(
                f"[tool {event.name} {status}] {json.dumps(event.output, default=str)}"
            )
    return "\n".join(lines)


@dataclass
class Turn:
    """One user message and the agent's response to it."""

    user_message: str
    response: TestResponse

    @property
    def agent_reply(self) -> str | None:
        return self.response.output

    @property
    def latency_ms(self) -> float:
        return self.response.duration_ms


@dataclass
class Trial:
    """A single simulated conversation and its judgement.

    Attributes:
        scenario: The (possibly reworded) scenario this conversation used.
        variation: Index of the variation, ``0`` for the scenario as written.
        repeat: Index of the repeat within the variation.
        turns: User/agent exchanges in order.
        verdicts: Judge verdict per success criterion.
        error: Why the conversation or judgement did not complete, if it did not.
        valid: ``False`` when infrastructure (simulated user or judge) failed,
            in which case the trial counts neither as a pass nor as a fail.
    """

    scenario: Scenario
    variation: int
    repeat: int
    turns: list[Turn] = field(default_factory=list)
    verdicts: dict[str, JudgeVerdict] = field(default_factory=dict)
    error: str | None = None
    valid: bool = True

    @property
    def transcript(self) -> list[RunEvent]:
        events: list[RunEvent] = []
        for turn in self.turns:
            events.append(ChatMessageEvent(role="user", content=turn.user_message))
            events.extend(turn.response.events)
        return events

    @property
    def tool_calls(self) -> list[FunctionCallEvent]:
        return [call for turn in self.turns for call in turn.response.function_calls]

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def latencies_ms(self) -> list[float]:
        return [turn.latency_ms for turn in self.turns]

    @property
    def passed(self) -> bool:
        return (
            self.valid
            and self.error is None
            and bool(self.verdicts)
            and all(v.success for v in self.verdicts.values())
        )

    def summary(self) -> str:
        """Multi-line description of the trial for assertion messages."""
        status = "INVALID" if not self.valid else ("PASS" if self.passed else "FAIL")
        lines = [
            f"Trial variation={self.variation} repeat={self.repeat}: {status} "
            f"({self.turn_count} turns)"
        ]
        if self.error:
            lines.append(f"  error: {self.error}")
        for criterion, verdict in self.verdicts.items():
            mark = "pass" if verdict.success else "fail"
            lines.append(f"  {criterion}: {mark} - {verdict.reason}")
        lines.append("  transcript:")
        lines.extend(
            f"    {line}" for line in render_transcript(self.transcript).splitlines()
        )
        return "\n".join(lines)


@dataclass
class SimulationResult:
    """Outcome of running a scenario: every trial plus aggregate pass metrics.

    Trials are ordered by variation, then repeat, so ``trials[0]`` is always
    the scenario as written.
    """

    scenario: Scenario
    trials: list[Trial]

    @property
    def k(self) -> int:
        """Number of repeats per variation the pass@k / pass^k metrics use."""
        return self.scenario.repeat

    @property
    def valid_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.valid]

    @property
    def invalid_trials(self) -> list[Trial]:
        return [t for t in self.trials if not t.valid]

    @property
    def passed(self) -> bool:
        """``True`` when at least one trial is valid and every valid trial passed."""
        valid = self.valid_trials
        return bool(valid) and all(t.passed for t in valid)

    @property
    def pass_rate(self) -> float | None:
        """Fraction of valid trials that passed, or ``None`` without valid trials."""
        valid = self.valid_trials
        if not valid:
            return None
        return sum(t.passed for t in valid) / len(valid)

    @property
    def pass_at_k(self) -> float | None:
        """Estimated chance at least one of ``k`` attempts at a variation passes.

        Computed per variation and averaged. ``None`` when any variation has
        fewer than ``k`` valid trials.
        """
        return self._mean_over_variations(pass_at_k)

    @property
    def pass_pow_k(self) -> float | None:
        """Estimated chance all ``k`` attempts at a variation pass.

        Computed per variation and averaged. ``None`` when any variation has
        fewer than ``k`` valid trials.
        """
        return self._mean_over_variations(pass_pow_k)

    def _mean_over_variations(
        self, estimator: Callable[[int, int, int], float]
    ) -> float | None:
        by_variation: dict[int, list[Trial]] = {}
        for trial in self.trials:
            by_variation.setdefault(trial.variation, []).append(trial)
        estimates: list[float] = []
        for group in by_variation.values():
            valid = [t for t in group if t.valid]
            if len(valid) < self.k:
                return None
            estimates.append(
                estimator(len(valid), sum(t.passed for t in valid), self.k)
            )
        if not estimates:
            return None
        return sum(estimates) / len(estimates)

    def summary(self) -> str:
        """Multi-line report suitable for an assertion message."""
        valid = self.valid_trials
        passed = sum(t.passed for t in valid)
        lines = [
            f"Scenario '{self.scenario.name}': {'PASS' if self.passed else 'FAIL'} "
            f"({passed}/{len(valid)} valid trials passed, "
            f"{len(self.invalid_trials)} invalid)"
        ]
        if self.pass_at_k is not None and self.pass_pow_k is not None:
            lines.append(
                f"pass@{self.k}={self.pass_at_k:.2f} pass^{self.k}={self.pass_pow_k:.2f}"
            )
        lines.extend(trial.summary() for trial in self.trials)
        return "\n".join(lines)


class Simulation:
    """Runs scenarios against an agent or LLM with an LLM-driven simulated user.

    Each conversation needs fresh LLM state, so when a scenario runs more
    than one conversation (``variations`` or ``repeat`` above 1) both the
    simulated-user LLM and the target must be given as factories.

    Args:
        user_llm: LLM that plays the user, or a factory that builds one.
        max_turns: Maximum user messages per conversation.
        turn_timeout: Seconds allowed for each user or agent turn.
    """

    def __init__(
        self,
        user_llm: LLM | Callable[[], LLM],
        max_turns: int = 10,
        turn_timeout: float = 60.0,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if turn_timeout <= 0:
            raise ValueError("turn_timeout must be positive")
        self._user_llm = user_llm
        self._max_turns = max_turns
        self._turn_timeout = turn_timeout

    async def run(
        self,
        agent_or_llm: Target | TargetFactory,
        scenario: Scenario,
        judge: Judge,
        instructions: str | None = None,
    ) -> SimulationResult:
        """Hold ``variations * repeat`` conversations and judge each one.

        Args:
            agent_or_llm: The agent or LLM under test, or a factory building one.
                An ``Agent`` supplies its own instructions.
            scenario: Scenario to simulate.
            judge: Judge that evaluates each success criterion over the transcript.
            instructions: System instructions for a bare ``LLM`` target.
                Ignored for an ``Agent``. Defaults to the ``TestSession`` default.

        Raises:
            ValueError: If more than one conversation is needed but an
                instance rather than a factory was given.
        """
        conversations = scenario.variations * scenario.repeat
        target_factory: TargetFactory
        if isinstance(agent_or_llm, (Agent, LLM)):
            _require_factory_for_many(conversations, "agent_or_llm")
            target_factory = _constant(agent_or_llm)
        else:
            target_factory = agent_or_llm
        user_factory: Callable[[], LLM]
        if isinstance(self._user_llm, LLM):
            _require_factory_for_many(conversations, "user_llm")
            user_factory = _constant(self._user_llm)
        else:
            user_factory = self._user_llm

        variants = [scenario]
        if scenario.variations > 1:
            variants = await generate_variations(
                user_factory(), scenario, scenario.variations
            )

        trials: list[Trial] = []
        for variation, variant in enumerate(variants):
            for repeat in range(scenario.repeat):
                trial = Trial(scenario=variant, variation=variation, repeat=repeat)
                await self._run_trial(
                    trial, target_factory(), user_factory(), judge, instructions
                )
                trials.append(trial)
                logger.info(
                    "Scenario %s variation=%d repeat=%d: %s",
                    scenario.name,
                    variation,
                    repeat,
                    "invalid"
                    if not trial.valid
                    else ("pass" if trial.passed else "fail"),
                )
        return SimulationResult(scenario=scenario, trials=trials)

    async def _run_trial(
        self,
        trial: Trial,
        target: Target,
        user_llm: LLM,
        judge: Judge,
        instructions: str | None,
    ) -> None:
        session_instructions: str | None
        if isinstance(target, Agent):
            llm = target.llm
            session_instructions = target.instructions.full_reference
        else:
            llm = target
            session_instructions = instructions

        user = SimulatedUser(
            user_llm,
            trial.scenario,
            max_turns=self._max_turns,
            turn_timeout=self._turn_timeout,
        )
        session = (
            TestSession(llm=llm, instructions=session_instructions)
            if session_instructions is not None
            else TestSession(llm=llm)
        )
        async with session:
            try:
                message = await user.next_message(None)
                while message is not None:
                    response = await asyncio.wait_for(
                        session.simple_response(message), timeout=self._turn_timeout
                    )
                    trial.turns.append(Turn(user_message=message, response=response))
                    message = await user.next_message(response.output or "")
            except asyncio.TimeoutError:
                trial.error = f"Agent did not reply within {self._turn_timeout}s"
                return
            except SimulatedUserError as exc:
                trial.error = str(exc)
                trial.valid = False
                return

        if not trial.turns:
            trial.error = (
                "Simulated user ended the conversation before sending a message"
            )
            trial.valid = False
            return

        transcript = ChatMessageEvent(
            role="assistant",
            content=(
                "Full conversation transcript between a user and the agent:\n\n"
                + render_transcript(trial.transcript)
            ),
        )
        for criterion in trial.scenario.success:
            intent = (
                f"Over the whole conversation the agent satisfied the success "
                f"criterion '{criterion}'.\nThe user's goal was: {trial.scenario.goal}"
            )
            try:
                trial.verdicts[criterion] = await judge.evaluate(transcript, intent)
            except JudgeError as exc:
                trial.error = f"Judge failed on '{criterion}': {exc}"
                trial.valid = False
                return
            except Exception as exc:
                logger.exception("Judge raised on criterion %r", criterion)
                trial.error = f"Judge failed on '{criterion}': {exc}"
                trial.valid = False
                return


def _constant(value: T) -> Callable[[], T]:
    return lambda: value


def _require_factory_for_many(conversations: int, name: str) -> None:
    if conversations > 1:
        raise ValueError(
            f"{name} must be a factory (a callable returning a new instance) "
            f"when a scenario runs {conversations} conversations"
        )
