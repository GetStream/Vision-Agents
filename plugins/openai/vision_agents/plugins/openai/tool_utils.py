"""Shared utilities for OpenAI tool/function calling."""

import json
from typing import Any, Dict, List

from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema


def convert_tools_to_openai_format(
    tools: List[ToolSchema], for_realtime: bool = False
) -> List[Dict[str, Any]]:
    """Convert ToolSchema to OpenAI format.

    Args:
        tools: List of ToolSchema objects from the function registry
        for_realtime: If True, format for Realtime API (no strict field).
                      If False, format for Responses API (includes strict).

    Returns:
        List of tools in OpenAI format
    """
    out = []
    for t in tools or []:
        params = t.get("parameters_schema") or t.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        params.setdefault("additionalProperties", False)

        tool_def: Dict[str, Any] = {
            "type": "function",
            "name": t.get("name", "unnamed_tool"),
            "description": t.get("description", "") or "",
            "parameters": params,
        }

        # Responses API supports strict mode, Realtime API does not
        if not for_realtime:
            tool_def["strict"] = True

        out.append(tool_def)
    return out


def tool_call_dedup_key(tc: NormalizedToolCallItem) -> tuple[str, str]:
    """Generate a deduplication key for a tool call.

    Uses only name and arguments, not id, so that logically identical
    tool calls are deduplicated even if they have different IDs.

    Args:
        tc: Normalized tool call item

    Returns:
        Tuple of (name, serialized_arguments) for deduplication
    """
    return (
        tc["name"],
        json.dumps(tc.get("arguments_json", {}), sort_keys=True),
    )


def parse_tool_arguments(args: str | dict) -> dict:
    """Parse tool arguments from string or dict.

    Args:
        args: Arguments as JSON string or dict

    Returns:
        Parsed arguments dict
    """
    if isinstance(args, dict):
        return args
    if not args:
        return {}
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return {}
