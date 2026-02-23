"""Intent evaluation for agent message testing.

Defines the ``Judge`` protocol and the default ``LLMJudge`` implementation
that uses a separate LLM instance with a structured PASS/FAIL prompt.
"""

import logging
from typing import Protocol, runtime_checkable

from vision_agents.core.llm.llm import LLM

logger = logging.getLogger(__name__)

_VERDICTS = {"PASS": True, "FAIL": False}
_ERROR_PREVIEW_MAX_LEN = 200

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict test evaluator for conversational AI agents.\n"
    "You will be shown a message produced by an agent and a target intent.\n"
    "Determine whether the message accomplishes the intent.\n\n"
    "Rules:\n"
    "- Be strict: if the message does not clearly fulfil the intent, it fails.\n"
    "- Respond with EXACTLY one line in one of these formats:\n"
    "  PASS: <brief reason>\n"
    "  FAIL: <brief reason>\n"
    "- Do NOT include any other text before or after the verdict line."
)


@runtime_checkable
class Judge(Protocol):
    """Evaluates whether an agent message fulfils a given intent."""

    async def evaluate(self, content: str, intent: str) -> tuple[bool, str]:
        """Return ``(success, reason)`` for *content* against *intent*."""
        ...


class LLMJudge:
    """Judge backed by an LLM instance.

    Uses a one-shot PASS/FAIL prompt to evaluate whether a message
    fulfils the given intent.

    Args:
        llm: LLM instance to use for evaluation.  Should be a
            **separate** instance from the agent's LLM so that
            judge calls do not pollute the agent's conversation.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def evaluate(self, content: str, intent: str) -> tuple[bool, str]:
        if not content:
            return False, "The message is empty."

        if not intent:
            return False, "Intent is required for evaluation."

        original_instructions = self._llm._instructions
        self._llm.set_instructions(_JUDGE_SYSTEM_PROMPT)

        try:
            prompt = (
                f"Check if the following message fulfils the given intent.\n\n"
                f"Intent:\n{intent}\n\n"
                f"Message:\n{content}\n\n"
                f"Respond with EXACTLY one line: PASS: <reason> or FAIL: <reason>"
            )

            response = await self._llm.simple_response(text=prompt)

            if not response or not response.text:
                return False, "LLM returned an empty response."

            return _parse_verdict(response.text)

        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("Judge evaluation failed")
            return False, f"Judge evaluation error: {exc}"

        finally:
            self._llm.set_instructions(original_instructions)


def _parse_verdict(text: str) -> tuple[bool, str]:
    """Parse a PASS/FAIL verdict from the LLM response text."""
    for line in text.strip().splitlines():
        word = line.strip().split(":")[0].strip().upper()
        if word in _VERDICTS:
            reason = line.strip()[len(word) :].lstrip(":").strip()
            return _VERDICTS[word], reason or f"{word.title()}ed."

    return (
        False,
        f"Could not parse verdict from LLM response: {text[:_ERROR_PREVIEW_MAX_LEN]}",
    )
