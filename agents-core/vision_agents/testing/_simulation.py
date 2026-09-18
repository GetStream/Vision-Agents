"""Multi-turn simulation runner: simulated user vs. agent, judged per criterion."""

import asyncio
import dataclasses
import importlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from math import comb
from typing import TypeVar

from vision_agents.core.agents.agents import Agent
from vision_agents.core.llm.llm import LLM
from vision_agents.core.stt.stt import STT
from vision_agents.core.tts.tts import TTS

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
from ._spoken import AgentSilentError, CallerError, SpokenConversation
from ._variations import generate_variations

logger = logging.getLogger(__name__)

T = TypeVar("T")

Target = Agent | LLM
TargetFactory = Callable[[], Target]
Caller = TypeVar("Caller", TTS, STT)


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
    """One user message and the agent's response to it.

    Attributes:
        user_message: What the simulated user said.
        response: What came back. In audio mode its output is what the
            caller's STT heard, which is what the judge reads.
        intended_reply: What the agent's LLM meant to say, in audio mode.
        voice_to_voice_ms: Time from the caller falling silent to the first
            agent audio with energy in it, in audio mode.

    ``latency_ms`` is the LLM's wall time in text mode and, in audio mode,
    the time from the caller falling silent to the end of the agent's reply
    as heard, STT lag included.
    """

    user_message: str
    response: TestResponse
    intended_reply: str | None = None
    voice_to_voice_ms: float | None = None

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
    def voice_to_voice_ms(self) -> list[float | None]:
        """Per-turn voice-to-voice latency; ``None`` per turn in a text conversation."""
        return [turn.voice_to_voice_ms for turn in self.turns]

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
        spoken = [t for t in self.turns if t.intended_reply is not None]
        if spoken:
            lines.append("  intended (what the agent meant to say):")
            lines.extend(f"    [assistant] {t.intended_reply}" for t in spoken)
            lines.append(
                "  voice-to-voice ms: "
                + ", ".join(
                    "n/a" if ms is None else f"{ms:.0f}"
                    for ms in self.voice_to_voice_ms
                )
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

    An ``audio`` scenario needs an ``Agent`` built with a ``LoopbackEdge``
    and a voice and ears for the caller: either factories passed here or
    plugin names in the scenario's ``caller_tts`` / ``caller_stt`` fields.

    Args:
        user_llm: LLM that plays the user, or a factory that builds one.
        max_turns: Maximum user messages per conversation.
        turn_timeout: Seconds allowed for each user or agent turn.
        caller_tts: TTS that speaks the user's lines in audio mode, or a
            factory that builds one. Overrides the scenario's ``caller_tts``.
        caller_stt: STT that transcribes the agent in audio mode, or a
            factory that builds one. Overrides the scenario's ``caller_stt``.
        audio_settle: Seconds of quiet after the agent's last sound before
            its reply counts as finished in audio mode.
    """

    def __init__(
        self,
        user_llm: LLM | Callable[[], LLM],
        max_turns: int = 10,
        turn_timeout: float = 60.0,
        caller_tts: TTS | Callable[[], TTS] | None = None,
        caller_stt: STT | Callable[[], STT] | None = None,
        audio_settle: float = 1.5,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if turn_timeout <= 0:
            raise ValueError("turn_timeout must be positive")
        if audio_settle <= 0:
            raise ValueError("audio_settle must be positive")
        self._user_llm = user_llm
        self._max_turns = max_turns
        self._turn_timeout = turn_timeout
        self._caller_tts = caller_tts
        self._caller_stt = caller_stt
        self._audio_settle = audio_settle

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
                instance rather than a factory was given, or if an audio
                scenario has no caller TTS or STT configured.
        """
        conversations = scenario.variations * scenario.repeat
        voice_factory: Callable[[], TTS] | None = None
        ears_factory: Callable[[], STT] | None = None
        if scenario.mode == "audio":
            voice_factory = _caller_factory(
                self._caller_tts,
                scenario.caller_tts,
                "TTS",
                conversations,
                scenario.name,
            )
            ears_factory = _caller_factory(
                self._caller_stt,
                scenario.caller_stt,
                "STT",
                conversations,
                scenario.name,
            )
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
                user = SimulatedUser(
                    user_factory(),
                    variant,
                    max_turns=self._max_turns,
                    turn_timeout=self._turn_timeout,
                )
                if voice_factory is not None and ears_factory is not None:
                    await self._converse_aloud(
                        trial, target_factory(), user, voice_factory(), ears_factory()
                    )
                else:
                    await self._converse_in_text(
                        trial, target_factory(), user, instructions
                    )
                await self._judge_trial(trial, judge)
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

    async def _converse_in_text(
        self,
        trial: Trial,
        target: Target,
        user: SimulatedUser,
        instructions: str | None,
    ) -> None:
        session_instructions: str | None
        if isinstance(target, Agent):
            llm = target.llm
            session_instructions = target.instructions.full_reference
        else:
            llm = target
            session_instructions = instructions

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

    async def _converse_aloud(
        self,
        trial: Trial,
        target: Target,
        user: SimulatedUser,
        voice: TTS,
        ears: STT,
    ) -> None:
        if not isinstance(target, Agent):
            raise ValueError(
                "Audio mode needs an Agent built with a LoopbackEdge "
                f"(vision_agents.testing.LoopbackEdge), got {type(target).__name__}"
            )
        conversation = SpokenConversation(
            target,
            voice,
            ears,
            turn_timeout=self._turn_timeout,
            settle=self._audio_settle,
        )
        try:
            async with conversation:
                message = await user.next_message(None)
                while message is not None:
                    started = time.monotonic()
                    line = await conversation.say(message)
                    events: list[RunEvent] = list(line.events)
                    events.append(
                        ChatMessageEvent(role="assistant", content=line.heard)
                    )
                    response = dataclasses.replace(
                        TestResponse.build(
                            events=events, user_input=message, start_time=started
                        ),
                        duration_ms=line.duration_ms,
                    )
                    trial.turns.append(
                        Turn(
                            user_message=message,
                            response=response,
                            intended_reply=line.intended,
                            voice_to_voice_ms=line.voice_to_voice_ms,
                        )
                    )
                    message = await user.next_message(line.heard)
        except AgentSilentError as exc:
            trial.error = str(exc)
        except (SimulatedUserError, CallerError) as exc:
            trial.error = str(exc)
            trial.valid = False

    async def _judge_trial(self, trial: Trial, judge: Judge) -> None:
        if trial.error is not None:
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


def _caller_factory(
    given: Caller | Callable[[], Caller] | None,
    plugin: str | None,
    kind: str,
    conversations: int,
    scenario_name: str,
) -> Callable[[], Caller]:
    """Pick the caller's voice or ears: what was passed in, else the scenario's plugin.

    ``kind`` is ``"TTS"`` or ``"STT"``; a plugin name resolves to
    ``vision_agents.plugins.<plugin>.<kind>`` built with its defaults.
    """
    field_name = f"caller_{kind.lower()}"
    if isinstance(given, (TTS, STT)):
        _require_factory_for_many(conversations, field_name)
        return _constant(given)
    if given is not None:
        return given
    if plugin is None:
        raise ValueError(
            f"Scenario {scenario_name!r} has mode 'audio' but no caller {kind} is "
            f"configured: pass {field_name} to Simulation or set {field_name} in "
            "the scenario"
        )
    try:
        module = importlib.import_module(f"vision_agents.plugins.{plugin}")
    except ImportError as exc:
        raise ValueError(
            f"Caller {kind} plugin {plugin!r} is not installed: "
            f"add vision-agents-plugins-{plugin} to your dependencies"
        ) from exc
    try:
        cls = module.TTS if kind == "TTS" else module.STT
    except AttributeError as exc:
        raise ValueError(f"Plugin {plugin!r} does not provide a {kind}") from exc

    def build() -> Caller:
        instance: Caller = cls()
        return instance

    return build


def _require_factory_for_many(conversations: int, name: str) -> None:
    if conversations > 1:
        raise ValueError(
            f"{name} must be a factory (a callable returning a new instance) "
            f"when a scenario runs {conversations} conversations"
        )
