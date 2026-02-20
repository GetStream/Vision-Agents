"""LLM-based intent evaluation for agent message testing.

Uses a separate LLM instance with a structured PASS/FAIL prompt to
evaluate whether an agent's message fulfils a given intent.
"""

import logging

from vision_agents.core.llm.llm import LLM

logger = logging.getLogger(__name__)

# Verdict keywords mapped to success/failure for _parse_verdict.
_VERDICTS = {"PASS": True, "FAIL": False}
_ERROR_PREVIEW_MAX_LEN = 200  # max chars of LLM response shown in parse errors

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


async def evaluate_intent(
    llm: LLM,
    message_content: str,
    intent: str,
) -> tuple[bool, str]:
    """Evaluate whether *message_content* fulfils the given *intent*.

    Creates a one-shot LLM call with a judge system prompt.  The LLM
    is expected to respond with a ``PASS: …`` or ``FAIL: …`` line.

    Important:
        The *llm* parameter should be a **separate** instance from the
        agent's LLM so that the judge call does not pollute the agent's
        conversation history.

    Args:
        llm: LLM instance to use for evaluation.
        message_content: The agent's message to evaluate.
        intent: Description of what the message should accomplish.

    Returns:
        ``(success, reason)`` tuple.
    """
    if not message_content:
        return False, "The message is empty."

    if not intent:
        return False, "Intent is required for evaluation."

    original_instructions = llm._instructions
    llm.set_instructions(_JUDGE_SYSTEM_PROMPT)

    try:
        prompt = (
            f"Check if the following message fulfils the given intent.\n\n"
            f"Intent:\n{intent}\n\n"
            f"Message:\n{message_content}\n\n"
            f"Respond with EXACTLY one line: PASS: <reason> or FAIL: <reason>"
        )

        response = await llm.simple_response(text=prompt)

        if not response or not response.text:
            return False, "LLM returned an empty response."

        return _parse_verdict(response.text)

    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("Judge evaluation failed")
        return False, f"Judge evaluation error: {exc}"

    finally:
        llm.set_instructions(original_instructions)


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
