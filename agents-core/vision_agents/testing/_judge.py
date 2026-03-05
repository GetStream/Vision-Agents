"""Intent evaluation for agent message testing.

Defines the ``Judge`` protocol and the default ``LLMJudge`` implementation
that uses a separate LLM instance with a structured JSON prompt.
"""

import json
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from vision_agents.core.llm.llm import LLM
from vision_agents.testing import ChatMessageEvent

logger = logging.getLogger(__name__)

_RESPONSE_PREVIEW_MAX_LEN = 200

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict test evaluator for conversational AI agents.\n"
    "You will be shown a message produced by an agent and a target intent.\n"
    "Determine whether the message accomplishes the intent.\n\n"
    "Rules:\n"
    "- Be strict: if the message does not clearly fulfil the intent, it fails.\n"
    "- Respond with ONLY a JSON object in this exact format:\n"
    '  {"verdict": "pass", "reason": "<brief reason>"}\n'
    '  {"verdict": "fail", "reason": "<brief reason>"}\n'
    "- Do NOT include any other text before or after the JSON."
)


@dataclass
class JudgeVerdict:
    """Result of a judge evaluation."""

    success: bool
    reason: str


@runtime_checkable
class Judge(Protocol):
    """Evaluates whether an agent message fulfils a given intent."""

    async def evaluate(self, event: ChatMessageEvent, intent: str) -> JudgeVerdict:
        """Return a verdict for *event* against *intent*."""
        ...


class LLMJudge:
    """Judge backed by an LLM instance.

    Uses a JSON prompt to evaluate whether a message fulfils the given
    intent.

    Args:
        llm: LLM instance to use for evaluation.  Should be a
            **separate** instance from the agent's LLM so that
            judge calls do not pollute the agent's conversation.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def evaluate(self, event: ChatMessageEvent, intent: str) -> JudgeVerdict:
        if not event.content:
            return JudgeVerdict(success=False, reason="The message is empty.")

        if not intent:
            return JudgeVerdict(
                success=False, reason="Intent is required for evaluation."
            )

        prompt = (
            f"Check if the following message fulfils the given intent.\n\n"
            f"Intent:\n{intent}\n\n"
            f"Message:\n{event.content}\n\n"
            'Respond with ONLY a JSON object: {"verdict": "pass" or "fail", "reason": "..."}'
        )

        original_instructions = self._llm._instructions
        self._llm.set_instructions(_JUDGE_SYSTEM_PROMPT)

        try:
            response = await self._llm.simple_response(text=prompt)

            if not response or not response.text:
                return JudgeVerdict(
                    success=False, reason="LLM returned an empty response."
                )

            return self._parse_verdict(response.text)

        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("Judge evaluation failed")
            return JudgeVerdict(success=False, reason=f"Judge evaluation error: {exc}")

        finally:
            self._llm.set_instructions(original_instructions)

    @staticmethod
    def _parse_verdict(text: str) -> JudgeVerdict:
        """Parse a JSON verdict from the LLM response text."""
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

        verdict = data.get("verdict", "").lower()
        reason = data.get("reason", "")

        if verdict == "pass":
            return JudgeVerdict(success=True, reason=reason or "Passed.")
        if verdict == "fail":
            return JudgeVerdict(success=False, reason=reason or "Failed.")

        return JudgeVerdict(
            success=False,
            reason=f"Unknown verdict '{verdict}' in LLM response: {text[:_RESPONSE_PREVIEW_MAX_LEN]}",
        )
