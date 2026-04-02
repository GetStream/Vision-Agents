"""Shared tool-calling utilities used by all HuggingFace plugin LLM implementations.

Provides the unified tool format conversion and the multi-round tool execution
loop that is shared across HuggingFaceLLM, HuggingFaceVLM, TransformersLLM,
and TransformersVLM.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema

logger = logging.getLogger(__name__)

FollowUpFn = Callable[
    [list[dict[str, Any]]],
    Awaitable[tuple[LLMResponseEvent, list[NormalizedToolCallItem]]],
]


def convert_tools_to_chat_completions_format(
    tools: list[ToolSchema],
) -> list[dict[str, Any]]:
    """Convert ToolSchema objects to the Chat Completions API tool format.

    This format is understood by both the HuggingFace Inference API and
    the ``tools`` kwarg of HuggingFace tokenizer / processor
    ``apply_chat_template`` methods.
    """
    result: list[dict[str, Any]] = []
    for t in tools or []:
        name = t.get("name", "unnamed_tool")
        description = t.get("description", "") or ""
        params = t.get("parameters_schema") or t.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": params,
                },
            }
        )
    return result


async def run_tool_call_loop(
    llm: LLM,
    tool_calls: list[NormalizedToolCallItem],
    messages: list[dict[str, Any]],
    generate_followup: FollowUpFn,
    *,
    max_rounds: int = 3,
    max_concurrency: int = 8,
    timeout_s: float = 30,
) -> LLMResponseEvent:
    """Execute tool calls and generate follow-up responses in a multi-round loop.

    Args:
        llm: The LLM instance (used for ``_dedup_and_execute`` and ``_sanitize_tool_output``).
        tool_calls: Initial tool calls to execute.
        messages: The conversation messages so far (will be mutated with tool results).
        generate_followup: Async callback that takes updated messages and returns
            ``(response, next_tool_calls)``.
        max_rounds: Maximum number of tool-call rounds before stopping.
        max_concurrency: Maximum concurrent tool executions per round.
        timeout_s: Timeout in seconds for each tool execution.
    """
    llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
    current_tool_calls = tool_calls
    seen: set[tuple[str | None, str, str]] = set()
    current_messages = list(messages)

    for round_num in range(max_rounds):
        logger.info(
            "Tool call round %d: executing %d call(s) — %s",
            round_num + 1,
            len(current_tool_calls),
            ", ".join(tc.get("name", "?") for tc in current_tool_calls),
        )

        triples, seen = await llm._dedup_and_execute(
            current_tool_calls,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
            seen=seen,
        )

        if not triples:
            break

        assistant_tool_calls = []
        tool_results = []
        for call_index, (tc, res, err) in enumerate(triples):
            cid = tc.get("id") or f"tool_call_{round_num}_{call_index}"

            name = tc["name"]
            args = tc.get("arguments_json", {})
            if err is not None:
                logger.warning("  [tool] %s(%s) failed: %s", name, args, err)
            else:
                logger.info("  [tool] %s(%s) → %s", name, args, res)

            assistant_tool_calls.append(
                {
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments_json", {})),
                    },
                }
            )
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": cid,
                    "content": llm._sanitize_tool_output(
                        err if err is not None else res
                    ),
                }
            )

        if not tool_results:
            return llm_response

        current_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            }
        )
        current_messages.extend(tool_results)

        follow_up, next_tool_calls = await generate_followup(current_messages)

        if next_tool_calls and round_num < max_rounds - 1:
            current_tool_calls = next_tool_calls
            llm_response = follow_up
            continue

        return follow_up

    return llm_response
