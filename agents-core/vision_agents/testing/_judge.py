"""Conversation evaluation for agent testing.

Defines the ``Judge`` protocol, ``Criterion`` with a set of built-in
criteria, and the default ``LLMJudge`` implementation that uses a separate
LLM instance with a structured JSON prompt.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from vision_agents.core.llm.llm import LLM

from ._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from ._utils import collect_simple_response

logger = logging.getLogger(__name__)

_RESPONSE_PREVIEW_MAX_LEN = 200
_TRANSCRIPT_VALUE_MAX_LEN = 2000

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict test evaluator for conversational AI agents.\n"
    "You will be shown a transcript of a conversation between a user and an "
    "agent, including the tool calls the agent made and their results, "
    "followed by a list of named criteria.\n"
    "Evaluate every criterion independently against the whole transcript.\n\n"
    "Rules:\n"
    "- Be strict: if the transcript does not clearly satisfy a criterion, it fails.\n"
    "- A tool call only counts as successful if a matching tool result is present "
    "and it is not a tool error.\n"
    "- Score each criterion from 0.0 (completely fails) to 1.0 (fully satisfies).\n"
    "- Respond with ONLY a JSON object in this exact format:\n"
    '  {"results": [{"name": "<criterion name>", "verdict": "pass" or "fail", '
    '"score": <0.0-1.0>, "reason": "<brief reason>"}]}\n'
    "- Include exactly one result per criterion, using the criterion names as given.\n"
    "- Do NOT include any other text before or after the JSON."
)


@dataclass
class Criterion:
    """A named condition a conversation is judged against."""

    name: str
    description: str


@dataclass
class CriterionVerdict:
    """Verdict for a single criterion."""

    name: str
    success: bool
    score: float
    reason: str


@dataclass
class JudgeVerdict:
    """Result of a judge evaluation.

    ``success`` is true only when every criterion passed and ``score`` is
    the mean criterion score in ``[0.0, 1.0]``.
    """

    success: bool
    reason: str
    score: float = 0.0
    criteria: list[CriterionVerdict] = field(default_factory=list)


SAY_DO_CONSISTENCY = Criterion(
    name="say_do_consistency",
    description=(
        "Every action or result the agent claims (looked something up, booked, "
        "sent, fetched data, etc.) is backed by a tool call in the transcript "
        "that returned successfully. The agent never reports tool results that "
        "were not actually returned and never claims success for a tool call "
        "that errored or was never made."
    ),
)

STAYS_IN_SCOPE = Criterion(
    name="stays_in_scope",
    description=(
        "The agent only addresses what the user asked and what its instructions "
        "cover. It does not drift into unrelated topics, volunteer unrelated "
        "information, or take on tasks outside its role."
    ),
)

CONCISE = Criterion(
    name="concise",
    description=(
        "Agent replies are brief and to the point: no filler, no repetition, no "
        "unnecessary preamble, and no restating of the user's question."
    ),
)

RESPONDS_IN_USER_LANGUAGE = Criterion(
    name="responds_in_user_language",
    description=(
        "Each agent reply is written in the same language the user used in the "
        "message it responds to."
    ),
)


@runtime_checkable
class Judge(Protocol):
    """Evaluates agent output against intents or criteria."""

    async def evaluate(self, event: ChatMessageEvent, intent: str) -> JudgeVerdict:
        """Return a verdict for a single message against *intent*."""
        ...

    async def evaluate_conversation(
        self,
        events: Sequence[RunEvent],
        criteria: Sequence[Criterion | str],
        *,
        instructions: str | None = None,
    ) -> JudgeVerdict:
        """Return a verdict for a whole conversation against *criteria*."""
        ...


class LLMJudge:
    """Judge backed by an LLM instance.

    Uses a JSON prompt to evaluate a conversation transcript against a set
    of criteria.

    Args:
        llm: LLM instance to use for evaluation.  Should be a
            **separate** instance from the agent's LLM so that
            judge calls do not pollute the agent's conversation.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self._llm.set_instructions(_JUDGE_SYSTEM_PROMPT)

    async def evaluate(self, event: ChatMessageEvent, intent: str) -> JudgeVerdict:
        """Evaluate a single message against one intent."""
        if not event.content:
            return JudgeVerdict(success=False, reason="The message is empty.")

        if not intent:
            return JudgeVerdict(
                success=False, reason="Intent is required for evaluation."
            )

        return await self.evaluate_conversation(
            [event], [Criterion(name="intent", description=intent)]
        )

    async def evaluate_conversation(
        self,
        events: Sequence[RunEvent],
        criteria: Sequence[Criterion | str],
        *,
        instructions: str | None = None,
    ) -> JudgeVerdict:
        """Evaluate a conversation transcript against each criterion.

        Args:
            events: The transcript, e.g. ``TestSession.transcript`` or
                ``TestResponse.events``.
            criteria: Criteria to check. A plain string is treated as an
                ad-hoc criterion whose name is the string itself.
            instructions: The agent's instructions, given to the judge as
                context (needed for scope-style criteria).

        Returns:
            ``JudgeVerdict`` with one ``CriterionVerdict`` per criterion.

        Raises:
            ValueError: If two criteria share the same name.
        """
        if not events:
            return JudgeVerdict(success=False, reason="The conversation is empty.")

        if not criteria:
            return JudgeVerdict(
                success=False, reason="At least one criterion is required."
            )

        resolved = [
            Criterion(name=c, description=c) if isinstance(c, str) else c
            for c in criteria
        ]
        names = [c.name for c in resolved]
        if len(set(names)) != len(names):
            raise ValueError(f"Criterion names must be unique, got {names!r}")

        prompt = self._build_prompt(events, resolved, instructions)

        try:
            _, response = await collect_simple_response(
                self._llm.simple_response(text=prompt)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("Judge evaluation failed")
            return JudgeVerdict(success=False, reason=f"Judge evaluation error: {exc}")

        if not response.text:
            return JudgeVerdict(success=False, reason="LLM returned an empty response.")

        return self._parse_verdict(response.text, resolved)

    @classmethod
    def _build_prompt(
        cls,
        events: Sequence[RunEvent],
        criteria: Sequence[Criterion],
        instructions: str | None,
    ) -> str:
        parts = ["Evaluate the conversation below against each criterion."]
        if instructions:
            parts.append(f"Agent instructions:\n{instructions}")
        parts.append(f"Transcript:\n{cls._format_transcript(events)}")
        criteria_lines = "\n".join(f"- {c.name}: {c.description}" for c in criteria)
        parts.append(f"Criteria:\n{criteria_lines}")
        parts.append(
            "Respond with ONLY the JSON object described in your instructions, "
            "with one result per criterion."
        )
        return "\n\n".join(parts)

    @classmethod
    def _format_transcript(cls, events: Sequence[RunEvent]) -> str:
        lines: list[str] = []
        for event in events:
            if isinstance(event, ChatMessageEvent):
                lines.append(f"[{event.role}] {event.content}")
            elif isinstance(event, FunctionCallEvent):
                call_id = f" {event.tool_call_id}" if event.tool_call_id else ""
                lines.append(
                    f"[tool call{call_id}] {event.name}({cls._dump(event.arguments)})"
                )
            elif isinstance(event, FunctionCallOutputEvent):
                call_id = f" {event.tool_call_id}" if event.tool_call_id else ""
                label = "tool error" if event.is_error else "tool result"
                lines.append(
                    f"[{label}{call_id}] {event.name} -> {cls._dump(event.output)}"
                )
        return "\n".join(lines)

    @staticmethod
    def _dump(value: object) -> str:
        text = json.dumps(value, default=str)
        if len(text) > _TRANSCRIPT_VALUE_MAX_LEN:
            return text[: _TRANSCRIPT_VALUE_MAX_LEN - 3] + "..."
        return text

    @classmethod
    def _parse_verdict(cls, text: str, criteria: Sequence[Criterion]) -> JudgeVerdict:
        """Parse the JSON results from the LLM response into a verdict."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.exception("Could not parse JSON from LLM response")
            return JudgeVerdict(
                success=False,
                reason=f"Could not parse JSON from LLM response: {text[:_RESPONSE_PREVIEW_MAX_LEN]}",
            )

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return JudgeVerdict(
                success=False,
                reason=f"Missing 'results' list in LLM response: {text[:_RESPONSE_PREVIEW_MAX_LEN]}",
            )

        by_name = {item.get("name"): item for item in results if isinstance(item, dict)}
        verdicts: list[CriterionVerdict] = []
        for criterion in criteria:
            raw = by_name.get(criterion.name)
            if raw is None:
                verdicts.append(
                    CriterionVerdict(
                        name=criterion.name,
                        success=False,
                        score=0.0,
                        reason="Judge returned no verdict for this criterion.",
                    )
                )
            else:
                verdicts.append(cls._parse_criterion(criterion.name, raw))

        success = all(v.success for v in verdicts)
        score = sum(v.score for v in verdicts) / len(verdicts)
        failing = [v for v in verdicts if not v.success]
        reason = "\n".join(f"{v.name}: {v.reason}" for v in (failing or verdicts))
        return JudgeVerdict(
            success=success, reason=reason, score=score, criteria=verdicts
        )

    @staticmethod
    def _parse_criterion(name: str, raw: dict[str, object]) -> CriterionVerdict:
        verdict = raw.get("verdict")
        verdict = verdict.lower() if isinstance(verdict, str) else ""
        reason = raw.get("reason")
        reason = reason if isinstance(reason, str) else ""

        if verdict not in ("pass", "fail"):
            return CriterionVerdict(
                name=name,
                success=False,
                score=0.0,
                reason=f"Unknown verdict '{verdict}' in LLM response.",
            )

        success = verdict == "pass"
        score_raw = raw.get("score")
        if isinstance(score_raw, (int, float)) and not isinstance(score_raw, bool):
            score = min(max(float(score_raw), 0.0), 1.0)
        else:
            score = 1.0 if success else 0.0

        return CriterionVerdict(
            name=name,
            success=success,
            score=score,
            reason=reason or ("Passed." if success else "Failed."),
        )
