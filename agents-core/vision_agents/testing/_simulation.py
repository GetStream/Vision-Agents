"""Text-mode agent simulation.

A caller LLM plays a scenario brief against a fresh agent, one turn at a time,
and a judge LLM rules on every criterion once the conversation ends. The
result types mirror the hosted ``SimulationRun`` schema so local and hosted
reports read the same way.
"""

import json
import logging
import statistics
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.llm import LLM
from vision_agents.core.llm.realtime import Realtime

from ._events import FunctionCallEvent, FunctionCallOutputEvent, RunEvent
from ._judge import parse_verdict
from ._scenario import Scenario
from ._session import TestSession
from ._utils import collect_simple_response, strip_code_fences

if TYPE_CHECKING:
    from vision_agents.core.agents import Agent

logger = logging.getLogger(__name__)

MODE = "text"
END_TOKEN = "[END]"
_PREVIEW_MAX_LEN = 200
_OPENING_PROMPT = "(The conversation starts. Send your first message.)"
_NO_REPLY = "(The agent did not reply with any text.)"

CALLER_INSTRUCTIONS = (
    "You are role-playing a person talking to an AI agent over text chat, "
    "one message at a time.\n\n"
    "Your brief:\n{brief}\n\n"
    "Rules:\n"
    "- Speak only as the person in the brief. Never speak for the agent and "
    "never mention that you are simulated.\n"
    "- Keep every message short and natural, the way a real person would.\n"
    "- Pursue the brief until it is resolved, the agent clearly cannot help, "
    "or the conversation has naturally ended.\n"
    "- When you are done, reply with exactly {end_token} and nothing else."
)

JUDGE_INSTRUCTIONS = (
    "You are a strict evaluator of conversations between a user and an AI agent.\n"
    "You will be shown a transcript and one criterion the agent had to meet.\n"
    "Decide whether the transcript clearly satisfies the criterion.\n\n"
    "Rules:\n"
    "- Be strict: if the transcript does not clearly satisfy the criterion, it fails.\n"
    "- Respond with ONLY a JSON object in this exact format:\n"
    '  {"verdict": "pass", "reason": "<brief reason>"}\n'
    '  {"verdict": "fail", "reason": "<brief reason>"}\n'
    "- Do NOT include any other text before or after the JSON."
)

VARIATIONS_INSTRUCTIONS = (
    "You rewrite briefs for people who test AI agents.\n"
    "Given a brief, write alternative phrasings that keep the same goal, facts "
    "and constraints but change the wording, tone, level of detail and order.\n"
    "Respond with ONLY a JSON array of strings, one per rewrite. "
    "Do NOT include any other text."
)

State = Literal["passed", "failed", "errored"]
Ended = Literal["complete", "turns", "failed"]

AgentFactory = Callable[[], Awaitable["Agent"]]
LLMFactory = Callable[[], LLM]


class SimulationError(RuntimeError):
    """The simulation could not reach a verdict."""


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
        caller: True when the simulated caller said it rather than the agent.
        text: What was said.
        at: When it was said.
        latency_ms: Agent lines only; time from the caller's message to the reply.
        tool_calls: Agent lines only; tools called while producing the reply.
    """

    caller: bool
    text: str
    at: datetime
    latency_ms: float | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller": self.caller,
            "text": self.text,
            "at": self.at.isoformat(),
            "latency_ms": self.latency_ms,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


@dataclass
class CriterionVerdict:
    """The judge's ruling on one criterion."""

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
        variation: Which phrasing of the brief this used; 0 is the brief as written.
        attempt: Which repeat of that variation this was, starting at 0.
        scenario: The wording this conversation used.
        state: ``passed`` or ``failed`` by the judge, ``errored`` when it never got that far.
        turns: How many times the caller spoke.
        ended: Why the conversation stopped: the caller finished, ``max_turns`` was hit, or it failed.
        criteria: The judge's ruling per criterion; empty when the case errored before judging.
        error: What went wrong, for an errored case.
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
    criteria: list[CriterionVerdict] = field(default_factory=list)
    error: str | None = None

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
    variations: int
    max_turns: int
    judge_target: str
    conversations: list[SimulationCase]
    started_at: datetime
    finished_at: datetime
    error: str | None = None
    mode: str = MODE

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
            "variations": self.variations,
            "max_turns": self.max_turns,
            "mode": self.mode,
            "judge_target": self.judge_target,
            "state": self.state,
            "cases": self.cases,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
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
    repeat: int
    started_at: datetime
    finished_at: datetime
    mode: str = MODE

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
            "mode": self.mode,
            "judge_target": self.judge_target,
            "repeat": self.repeat,
            "cases": self.cases,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "runs": [run.to_dict() for run in self.runs],
        }


def format_transcript(transcript: list[TranscriptLine]) -> str:
    """Render a transcript as ``User:`` / ``Agent:`` lines for a judge."""
    return "\n".join(
        f"{'User' if line.caller else 'Agent'}: {line.text}" for line in transcript
    )


class Simulator:
    """Runs scenarios against an agent in text mode.

    Every conversation gets a fresh agent from ``create_agent`` so no history
    leaks between cases. ``llm_factory`` supplies the LLMs that play the
    caller, rewrite the brief for variations and judge the transcript; it is
    called once per role so their instructions never mix.

    Args:
        create_agent: Coroutine function returning a new, ready-to-use agent.
        llm_factory: Returns a fresh LLM instance on every call.
        judge_target: Human-readable name of the judge model, kept in the report.
        repeat: How many times to run every variation.
        variations: Overrides ``Scenario.variations`` for every scenario when set.
    """

    def __init__(
        self,
        create_agent: AgentFactory,
        llm_factory: LLMFactory,
        judge_target: str,
        *,
        repeat: int = 1,
        variations: int | None = None,
    ) -> None:
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        if variations is not None and variations < 1:
            raise ValueError("variations must be >= 1")
        self._create_agent = create_agent
        self._llm_factory = llm_factory
        self._judge_target = judge_target
        self._repeat = repeat
        self._variations = variations

    async def run(
        self,
        scenarios: list[Scenario],
        on_case: Callable[[Scenario, SimulationCase], None] | None = None,
    ) -> SimulationReport:
        """Run every scenario in order and collect the report.

        Args:
            scenarios: Scenarios to run.
            on_case: Called after every conversation, for progress output.
        """
        started = _now()
        runs = [await self.run_scenario(scenario, on_case) for scenario in scenarios]
        return SimulationReport(
            runs=runs,
            judge_target=self._judge_target,
            repeat=self._repeat,
            started_at=started,
            finished_at=_now(),
        )

    async def run_scenario(
        self,
        scenario: Scenario,
        on_case: Callable[[Scenario, SimulationCase], None] | None = None,
    ) -> SimulationRun:
        """Run one scenario: every variation, ``repeat`` times each."""
        started = _now()
        variations = self._variations or scenario.variations
        conversations: list[SimulationCase] = []
        error: str | None = None
        try:
            briefs = await self._briefs(scenario.scenario, variations)
        except Exception as exc:
            logger.exception("Failed to generate variations for %s", scenario.name)
            error = f"failed to generate variations: {exc}"
            briefs = []

        for variation, brief in enumerate(briefs):
            for attempt in range(self._repeat):
                case = await self._run_case(scenario, variation, attempt, brief)
                conversations.append(case)
                if on_case is not None:
                    on_case(scenario, case)

        return SimulationRun(
            id=uuid.uuid4().hex,
            name=scenario.name,
            scenario=scenario.scenario,
            criteria=list(scenario.criteria),
            variations=variations,
            max_turns=scenario.max_turns,
            judge_target=self._judge_target,
            conversations=conversations,
            error=error,
            started_at=started,
            finished_at=_now(),
        )

    async def _briefs(self, brief: str, variations: int) -> list[str]:
        """The brief as written, followed by ``variations - 1`` rewrites of it."""
        if variations == 1:
            return [brief]
        wanted = variations - 1
        llm = self._llm_factory()
        llm.set_instructions(VARIATIONS_INSTRUCTIONS)
        prompt = (
            f"Write {wanted} alternative phrasings of this brief.\n\n"
            f"Brief:\n{brief}\n\n"
            f"Respond with ONLY a JSON array of {wanted} strings."
        )
        try:
            _, final = await collect_simple_response(llm.simple_response(text=prompt))
        finally:
            await llm.close()
        rewrites = _parse_string_list(final.text)
        if len(rewrites) < wanted:
            raise SimulationError(
                f"asked for {wanted} variations but the model returned {len(rewrites)}"
            )
        return [brief, *rewrites[:wanted]]

    async def _run_case(
        self, scenario: Scenario, variation: int, attempt: int, brief: str
    ) -> SimulationCase:
        started = _now()
        transcript: list[TranscriptLine] = []
        verdicts: list[CriterionVerdict] = []
        ended: Ended = "failed"
        error: str | None = None
        try:
            agent = await self._create_agent()
            try:
                if isinstance(agent.llm, Realtime):
                    raise SimulationError(
                        "the agent uses a Realtime LLM, which needs an audio session; "
                        "simulate runs in text mode and needs a text LLM"
                    )
                async with TestSession(
                    llm=agent.llm, instructions=agent.instructions.full_reference
                ) as session:
                    ended = await self._converse(
                        session, brief, scenario.max_turns, transcript
                    )
                verdicts = await self._judge(transcript, scenario.criteria)
            finally:
                await agent.close()
        except Exception as exc:
            logger.exception(
                "Scenario %s variation %d attempt %d errored",
                scenario.name,
                variation,
                attempt,
            )
            error = str(exc)

        state: State
        if error is not None:
            state = "errored"
        elif all(v.passed for v in verdicts):
            state = "passed"
        else:
            state = "failed"

        return SimulationCase(
            id=uuid.uuid4().hex,
            variation=variation,
            attempt=attempt,
            scenario=brief,
            state=state,
            transcript=transcript,
            turns=sum(1 for line in transcript if line.caller),
            ended=ended,
            criteria=verdicts,
            error=error,
            started_at=started,
            finished_at=_now(),
        )

    async def _converse(
        self,
        session: TestSession,
        brief: str,
        max_turns: int,
        transcript: list[TranscriptLine],
    ) -> Ended:
        """Alternate caller and agent until the caller is done or turns run out."""
        caller = _Caller(self._llm_factory(), brief)
        try:
            text, done = _split_end_token(await caller.say(_OPENING_PROMPT))
            turns = 0
            while True:
                if not text:
                    if done and turns > 0:
                        return "complete"
                    raise SimulationError(
                        "the caller model ended before saying anything"
                        if done
                        else "the caller model returned an empty message"
                    )
                turns += 1
                transcript.append(TranscriptLine(caller=True, text=text, at=_now()))
                response = await session.simple_response(text)
                reply = response.output or ""
                transcript.append(
                    TranscriptLine(
                        caller=False,
                        text=reply,
                        at=_now(),
                        latency_ms=response.duration_ms,
                        tool_calls=_tool_calls(response.events),
                    )
                )
                if done:
                    return "complete"
                if turns >= max_turns:
                    return "turns"
                text, done = _split_end_token(await caller.say(reply or _NO_REPLY))
        finally:
            await caller.close()

    async def _judge(
        self, transcript: list[TranscriptLine], criteria: list[str]
    ) -> list[CriterionVerdict]:
        """Ask a fresh judge LLM about every criterion in turn."""
        rendered = format_transcript(transcript) or "(empty conversation)"
        verdicts: list[CriterionVerdict] = []
        for criterion in criteria:
            llm = self._llm_factory()
            llm.set_instructions(JUDGE_INSTRUCTIONS)
            prompt = (
                f"Criterion:\n{criterion}\n\n"
                f"Transcript:\n{rendered}\n\n"
                'Respond with ONLY a JSON object: {"verdict": "pass" or "fail", "reason": "..."}'
            )
            try:
                _, final = await collect_simple_response(
                    llm.simple_response(text=prompt)
                )
            finally:
                await llm.close()
            if not final.text:
                raise SimulationError(
                    f"judge returned an empty response for criterion {criterion!r}"
                )
            verdict = parse_verdict(final.text)
            verdicts.append(
                CriterionVerdict(
                    criterion=criterion, passed=verdict.success, reason=verdict.reason
                )
            )
        return verdicts


class _Caller:
    """Plays the person described in a brief, one message per turn."""

    def __init__(self, llm: LLM, brief: str) -> None:
        instructions = CALLER_INSTRUCTIONS.format(brief=brief, end_token=END_TOKEN)
        self._llm = llm
        self._llm.set_instructions(instructions)
        self._conversation = InMemoryConversation(
            instructions=instructions, messages=[]
        )
        self._llm.set_conversation(self._conversation)

    async def say(self, heard: str) -> str:
        """Feed the caller what the agent said and return the caller's next message."""
        await self._conversation.send_message(
            role="user", user_id="agent", content=heard
        )
        _, final = await collect_simple_response(self._llm.simple_response(text=heard))
        await self._conversation.send_message(
            role="assistant", user_id="caller", content=final.text
        )
        return final.text

    async def close(self) -> None:
        await self._llm.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _split_end_token(text: str) -> tuple[str, bool]:
    """Strip the end token and report whether the caller is done."""
    return text.replace(END_TOKEN, "").strip(), END_TOKEN in text


def _tool_calls(events: list[RunEvent]) -> list[ToolCall]:
    """Pair function calls with their outputs by name, in order."""
    calls: list[ToolCall] = []
    pending: list[ToolCall] = []
    for event in events:
        if isinstance(event, FunctionCallEvent):
            call = ToolCall(name=event.name, arguments=event.arguments)
            calls.append(call)
            pending.append(call)
        elif isinstance(event, FunctionCallOutputEvent):
            for call in pending:
                if call.name == event.name:
                    call.output = event.output
                    call.is_error = event.is_error
                    pending.remove(call)
                    break
    return calls


def _parse_string_list(text: str) -> list[str]:
    try:
        data = json.loads(strip_code_fences(text))
    except json.JSONDecodeError as exc:
        raise SimulationError(
            f"could not parse variations as JSON: {text[:_PREVIEW_MAX_LEN]}"
        ) from exc
    if not isinstance(data, list) or not all(
        isinstance(item, str) and item.strip() for item in data
    ):
        raise SimulationError(
            f"expected a JSON array of strings for variations: {text[:_PREVIEW_MAX_LEN]}"
        )
    return [item.strip() for item in data]
